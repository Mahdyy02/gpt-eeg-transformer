# train_seizure_prediction.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import mne
from pathlib import Path
import pandas as pd
from collections import defaultdict

"""
Seizure Prediction Transformer
- Input: 30-second EEG segment (raw, unfiltered)
- Output: Binary classification (seizure within next 30 seconds?)
- Uses transformer encoder architecture with classification head
"""

# ---------------- Hyperparameters ----------------
batch_size = 16
accumulation_steps = 4
segment_duration = 30.0  # seconds
prediction_horizon = 30.0  # seconds (predict seizure within next 30s)
sampling_rate = 250  # Hz (will resample if needed)
checkpoint_path = "seizure_prediction_finetuned.pth"
pretrained_checkpoint = "eeg_transformer.pth"  # Pre-trained model
max_iters = 5000  # Reduced for fine-tuning
eval_interval = 200  # More frequent evaluation
learning_rate = 3e-5  # Lower LR for fine-tuning
freeze_backbone = False  # Set to True to freeze transformer layers
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 20
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.2
max_seq_len = int(segment_duration * sampling_rate)  # 7500 samples
use_gradient_checkpointing = True
class_weights = None  # Will be computed from data
# -------------------------------------------------

print(f"Device: {device}")
print(f"Segment duration: {segment_duration}s ({max_seq_len} samples @ {sampling_rate}Hz)")
print(f"Prediction horizon: {prediction_horizon}s")
torch.manual_seed(1337)
if device == 'cuda':
    torch.cuda.manual_seed_all(1337)

# ---------------- Data Loading ----------------
def parse_annotations(csv_path):
    """
    Parse CSV annotations to get seizure time windows.
    Returns dict: {channel: [(start, stop, label), ...]}
    """
    df = pd.read_csv(csv_path, comment='#')
    annotations = defaultdict(list)
    for _, row in df.iterrows():
        channel = row['channel']
        start = float(row['start_time'])
        stop = float(row['stop_time'])
        label = row['label']
        annotations[channel].append((start, stop, label))
    return annotations


def get_seizure_windows(annotations):
    """
    Extract all seizure time windows across all channels.
    Returns list of (start, stop) tuples in seconds.
    """
    seizure_labels = {'fnsz', 'gnsz', 'spsz', 'cpsz', 'absz', 'tnsz', 'tcsz', 'mysz'}
    seizure_windows = []
    
    # Collect all seizure events from all channels
    for channel, events in annotations.items():
        for start, stop, label in events:
            if label in seizure_labels:
                seizure_windows.append((start, stop))
    
    # Merge overlapping windows
    if not seizure_windows:
        return []
    
    seizure_windows.sort()
    merged = [seizure_windows[0]]
    for start, stop in seizure_windows[1:]:
        if start <= merged[-1][1]:  # Overlapping
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    
    return merged


def is_seizure_in_window(time_start, time_end, seizure_windows):
    """Check if any seizure occurs within [time_start, time_end]."""
    for sz_start, sz_stop in seizure_windows:
        # Check if seizure overlaps with window
        if sz_start < time_end and sz_stop > time_start:
            return True
    return False


def load_eeg_session(edf_path, csv_path, segment_duration, prediction_horizon, sampling_rate, target_channels=None):
    """
    Load one EEG session and create labeled segments.
    Returns:
        segments: list of (signal_array, label) where signal is (n_channels, seq_len)
    """
    # Load EDF
    print(f"  Loading {edf_path.name}...")
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    
    # Resample if needed
    if raw.info['sfreq'] != sampling_rate:
        raw.resample(sampling_rate, verbose=False)
    
    # Get data: (n_channels, n_samples)
    data = raw.get_data()
    
    # Handle variable channel counts
    if target_channels is not None:
        if data.shape[0] < target_channels:
            # Pad with zeros
            padding = np.zeros((target_channels - data.shape[0], data.shape[1]))
            data = np.vstack([data, padding])
        elif data.shape[0] > target_channels:
            # Truncate to target
            data = data[:target_channels, :]
    
    duration = data.shape[1] / sampling_rate
    
    # Parse annotations
    annotations = parse_annotations(csv_path)
    seizure_windows = get_seizure_windows(annotations)
    
    print(f"    Duration: {duration:.1f}s, Found {len(seizure_windows)} seizure windows")
    
    # Create segments
    segment_samples = int(segment_duration * sampling_rate)
    horizon_samples = int(prediction_horizon * sampling_rate)
    
    segments = []
    pos_count = 0
    neg_count = 0
    
    # Slide through the recording
    step_size = segment_samples // 2  # 50% overlap
    for start_sample in range(0, data.shape[1] - segment_samples - horizon_samples, step_size):
        # Extract segment
        segment = data[:, start_sample:start_sample + segment_samples]
        
        # Time window for prediction
        segment_end_time = (start_sample + segment_samples) / sampling_rate
        prediction_end_time = segment_end_time + prediction_horizon
        
        # Label: is there a seizure in the next 30 seconds?
        label = is_seizure_in_window(segment_end_time, prediction_end_time, seizure_windows)
        
        segments.append((segment, int(label)))
        
        if label:
            pos_count += 1
        else:
            neg_count += 1
    
    print(f"    Created {len(segments)} segments: {pos_count} positive, {neg_count} negative")
    return segments


def load_all_sessions(data_dir, segment_duration, prediction_horizon, sampling_rate):
    """Load all EEG sessions from aaaaadpj patient."""
    data_dir = Path(data_dir)
    all_segments = []
    
    print("Loading EEG sessions...")
    
    # First pass: determine maximum channel count
    max_channels = 0
    session_files = []
    
    # s003_2006
    session_dir = data_dir / "s003_2006" / "02_tcp_le"
    edf_path = session_dir / "aaaaadpj_s003_t000.edf"
    csv_path = session_dir / "aaaaadpj_s003_t000.csv"
    if edf_path.exists() and csv_path.exists():
        session_files.append((edf_path, csv_path))
        raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)
        max_channels = max(max_channels, len(raw.ch_names))
    
    # s005_2006 (t000 to t005)
    session_dir = data_dir / "s005_2006" / "03_tcp_ar_a"
    for i in range(6):
        edf_path = session_dir / f"aaaaadpj_s005_t00{i}.edf"
        csv_path = session_dir / f"aaaaadpj_s005_t00{i}.csv"
        if edf_path.exists() and csv_path.exists():
            session_files.append((edf_path, csv_path))
            raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)
            max_channels = max(max_channels, len(raw.ch_names))
    
    print(f"Maximum channel count across sessions: {max_channels}")
    print(f"All segments will be standardized to {max_channels} channels\n")
    
    # Second pass: load with standardized channel count
    for edf_path, csv_path in session_files:
        segments = load_eeg_session(edf_path, csv_path, segment_duration, prediction_horizon, sampling_rate, target_channels=max_channels)
        all_segments.extend(segments)
    
    # Compute class distribution
    pos = sum(1 for _, label in all_segments if label == 1)
    neg = len(all_segments) - pos
    print(f"\n{'='*60}")
    print(f"Total segments: {len(all_segments)}")
    print(f"Positive (seizure): {pos} ({100*pos/len(all_segments):.1f}%)")
    print(f"Negative (no seizure): {neg} ({100*neg/len(all_segments):.1f}%)")
    print(f"{'='*60}\n")
    
    return all_segments, pos, neg


# ---------------- Normalization ----------------
def normalize_segment(segment):
    """
    Z-score normalization per channel.
    segment: (n_channels, seq_len)
    Returns: normalized segment
    """
    mean = segment.mean(axis=1, keepdims=True)
    std = segment.std(axis=1, keepdims=True) + 1e-8
    return (segment - mean) / std


# ---------------- Dataset ----------------
class SeizureDataset(torch.utils.data.Dataset):
    def __init__(self, segments):
        """
        segments: list of (signal, label) tuples
        signal: (n_channels, seq_len) numpy array
        label: int (0 or 1)
        """
        self.segments = segments
        
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        signal, label = self.segments[idx]
        
        # Normalize
        signal = normalize_segment(signal)
        
        # Convert to torch
        signal = torch.from_numpy(signal).float()  # (n_channels, seq_len)
        label = torch.tensor(label, dtype=torch.long)
        
        return signal, label


# ---------------- Model Architecture ----------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (B, T, C)
        return x + self.pe[:x.size(1), :].unsqueeze(0)


class TransformerEncoderFineTune(nn.Module):
    def __init__(self, n_channels, n_embd, n_head, n_layer, max_seq_len, dropout, vocab_size, pretrained_path=None):
        super().__init__()
        self.n_channels = n_channels
        self.n_embd = n_embd
        self.vocab_size = vocab_size
        
        # Load pre-trained transformer components
        if pretrained_path and Path(pretrained_path).exists():
            print(f"\n{'='*60}")
            print(f"Loading pre-trained transformer from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            pretrained_block_size = checkpoint['hyperparameters']['block_size']
            
            # Token embedding from pre-trained model (will be adapted)
            self.token_embedding = nn.Embedding(vocab_size, n_embd)
            self.token_embedding.weight.data.copy_(checkpoint['model_state_dict']['token_embedding_table.weight'])
            
            # Position embedding from pre-trained model (will be extended if needed)
            self.position_embedding = nn.Embedding(max_seq_len, n_embd)
            pretrained_pos_weight = checkpoint['model_state_dict']['position_embedding_table.weight']
            if max_seq_len <= pretrained_block_size:
                self.position_embedding.weight.data.copy_(pretrained_pos_weight[:max_seq_len])
            else:
                # Extend position embeddings by repeating
                self.position_embedding.weight.data[:pretrained_block_size].copy_(pretrained_pos_weight)
                for i in range(pretrained_block_size, max_seq_len):
                    self.position_embedding.weight.data[i] = pretrained_pos_weight[i % pretrained_block_size]
            
            print(f"✓ Loaded token embeddings (vocab_size={vocab_size})")
            print(f"✓ Loaded position embeddings (extended to {max_seq_len})")
            print(f"{'='*60}\n")
        else:
            # Initialize from scratch if no pretrained model
            print("\n⚠️  No pre-trained model found, initializing from scratch\n")
            self.token_embedding = nn.Embedding(vocab_size, n_embd)
            self.position_embedding = nn.Embedding(max_seq_len, n_embd)
            self.apply(self._init_weights)
        
        # Project channels to embedding dimension (NEW - not in pretrained model)
        self.channel_proj = nn.Linear(n_channels, n_embd)
        
        # Positional encoding for continuous signals
        self.pos_encoder = PositionalEncoding(n_embd, max_len=max_seq_len)
        
        # Transformer encoder layers (adapted from pretrained decoder)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=4 * n_embd,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        
        # NEW: Classification head for seizure prediction
        self.classifier = nn.Sequential(
            nn.LayerNorm(n_embd),
            nn.Linear(n_embd, n_embd),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_embd, 2)  # Binary classification
        )
        
        # Initialize new components
        self._init_new_weights()
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _init_new_weights(self):
        """Initialize only the new components (channel_proj and classifier)"""
        for module in [self.channel_proj, self.classifier]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        x: (B, n_channels, seq_len)
        Returns: logits (B, 2)
        """
        B, C, T = x.shape
        
        # Transpose to (B, T, C)
        x = x.transpose(1, 2)  # (B, seq_len, n_channels)
        
        # Project channels to embedding
        x = self.channel_proj(x)  # (B, seq_len, n_embd)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder (leverages pre-trained patterns)
        x = self.transformer_encoder(x)  # (B, seq_len, n_embd)
        
        # Global average pooling over time
        x = x.mean(dim=1)  # (B, n_embd)
        
        # Classification
        logits = self.classifier(x)  # (B, 2)
        
        return logits
    
    def freeze_backbone(self):
        """Freeze transformer encoder layers for feature extraction only"""
        for param in self.transformer_encoder.parameters():
            param.requires_grad = False
        print("✓ Transformer encoder frozen (feature extraction mode)")
    
    def unfreeze_backbone(self):
        """Unfreeze transformer encoder for full fine-tuning"""
        for param in self.transformer_encoder.parameters():
            param.requires_grad = True
        print("✓ Transformer encoder unfrozen (fine-tuning mode)")


# ---------------- Training ----------------
def train_model(model, train_loader, val_loader, optimizer, scaler, class_weights, vocab_size):
    """Training loop."""
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Loss function with class weights
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}\n")
    
    global_step = 0
    for epoch in range(max_iters // len(train_loader) + 1):
        if global_step >= max_iters:
            break
        
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        
        optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, (signals, labels) in enumerate(train_loader):
            if global_step >= max_iters:
                break
            
            signals = signals.to(device)
            labels = labels.to(device)
            
            # Forward pass
            with autocast():
                logits = model(signals)
                loss = criterion(logits, labels) / accumulation_steps
            
            # Backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            # Metrics
            epoch_loss += loss.item() * accumulation_steps
            preds = logits.argmax(dim=1)
            epoch_correct += (preds == labels).sum().item()
            epoch_total += labels.size(0)
            
            global_step += 1
            
            # Evaluation
            if global_step % eval_interval == 0:
                train_acc = epoch_correct / epoch_total if epoch_total > 0 else 0
                val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate_model(model, val_loader, criterion)
                
                train_losses.append(epoch_loss / (batch_idx + 1))
                val_losses.append(val_loss)
                train_accs.append(train_acc)
                val_accs.append(val_acc)
                
                print(f"\nStep {global_step}:")
                print(f"  Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc:.4f}")
                print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                print(f"  Val Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
                
                # Save checkpoint
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step': global_step,
                    'vocab_size': vocab_size,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'train_accs': train_accs,
                    'val_accs': val_accs,
                    'hyperparameters': {
                        'n_channels': model.n_channels,
                        'n_embd': n_embd,
                        'n_head': n_head,
                        'n_layer': n_layer,
                        'max_seq_len': max_seq_len,
                        'dropout': dropout,
                        'segment_duration': segment_duration,
                        'prediction_horizon': prediction_horizon,
                        'sampling_rate': sampling_rate
                    }
                }, checkpoint_path)
                print(f"  Checkpoint saved to {checkpoint_path}")
                
                model.train()
    
    return train_losses, val_losses, train_accs, val_accs


@torch.no_grad()
def evaluate_model(model, val_loader, criterion):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for signals, labels in val_loader:
        signals = signals.to(device)
        labels = labels.to(device)
        
        logits = model(signals)
        loss = criterion(logits, labels)
        
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Metrics
    accuracy = (all_preds == all_labels).mean()
    
    # Precision, Recall, F1 for positive class
    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return total_loss / len(val_loader), accuracy, precision, recall, f1


# ---------------- Main ----------------
def main():
    # Load data
    data_dir = Path("aaaaadpj")
    all_segments, pos_count, neg_count = load_all_sessions(
        data_dir, segment_duration, prediction_horizon, sampling_rate
    )
    
    # Compute class weights (inverse frequency)
    total = len(all_segments)
    weight_pos = total / (2 * pos_count) if pos_count > 0 else 1.0
    weight_neg = total / (2 * neg_count) if neg_count > 0 else 1.0
    class_weights = torch.tensor([weight_neg, weight_pos], dtype=torch.float32)
    print(f"Class weights: [neg={weight_neg:.2f}, pos={weight_pos:.2f}]\n")
    
    # Split into train/val (80/20)
    np.random.seed(1337)
    indices = np.random.permutation(len(all_segments))
    split_idx = int(0.8 * len(all_segments))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_segments = [all_segments[i] for i in train_indices]
    val_segments = [all_segments[i] for i in val_indices]
    
    print(f"Train segments: {len(train_segments)}")
    print(f"Val segments: {len(val_segments)}\n")
    
    # Create datasets
    train_dataset = SeizureDataset(train_segments)
    val_dataset = SeizureDataset(val_segments)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )
    
    # Get number of channels from first sample
    n_channels = train_segments[0][0].shape[0]
    print(f"Number of EEG channels: {n_channels}\n")
    
    # Load vocabulary size from pretrained model
    vocab_size = 512  # Default
    if Path(pretrained_checkpoint).exists():
        checkpoint = torch.load(pretrained_checkpoint, map_location='cpu')
        vocab_size = checkpoint['vocab_size']
        print(f"Pre-trained model vocab size: {vocab_size}")
    
    # Create model with pre-trained weights
    model = TransformerEncoderFineTune(
        n_channels=n_channels,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        max_seq_len=max_seq_len,
        dropout=dropout,
        vocab_size=vocab_size,
        pretrained_path=pretrained_checkpoint
    ).to(device)
    
    # Optionally freeze backbone
    if freeze_backbone:
        model.freeze_backbone()
    
    print(f"Model has {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params/1e6:.2f}M")
    
    # Optimizer (only for trainable parameters)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=learning_rate, 
        weight_decay=0.01
    )
    
    # Scaler for mixed precision
    if device == 'cuda':
        scaler = GradScaler()
    else:
        scaler = None
    
    # Train
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, optimizer, scaler, class_weights, vocab_size
    )
    
    # Plot results
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"{'='*60}\n")
    
    # Plot losses
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    steps = [i * eval_interval for i in range(len(train_losses))]
    
    axes[0].plot(steps, train_losses, label='Train Loss', marker='o', linewidth=2)
    axes[0].plot(steps, val_losses, label='Val Loss', marker='s', linewidth=2)
    axes[0].set_xlabel('Step', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(steps, train_accs, label='Train Accuracy', marker='o', linewidth=2)
    axes[1].plot(steps, val_accs, label='Val Accuracy', marker='s', linewidth=2)
    axes[1].set_xlabel('Step', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('seizure_prediction_training.png', dpi=150)
    print("Training plot saved to seizure_prediction_training.png")
    plt.show()
    
    print("\n✅ Training complete! Checkpoint saved.")


if __name__ == "__main__":
    main()
