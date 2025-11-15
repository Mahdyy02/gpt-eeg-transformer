# train_eeg_transformer_full_fixed.py
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import trange
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import mne
from pathlib import Path
from contextlib import nullcontext

"""
Full corrected EEG transformer training script implementing:
- single-sample quantization with 1024 levels
- normalization before quantization
- dual loss: CrossEntropy + MSE on dequantized waveform
- larger context (block_size)
- transformer: n_embd=256, n_layer=6, n_head=8
- gradient accumulation (batch_size=64, accumulation_steps=4)
- max_samples cap to limit dataset size for quick experiments
"""

# ---------------- Hyperparameters (OPTIMIZED FOR MEMORY) ----------------
batch_size = 16                   # Reduced from 64
accumulation_steps = 8            # Increased to maintain effective batch size
block_size = 512                  # Reduced from 2048
checkpoint_path = "eeg_transformer_memory_opt.pth"
max_iters = 30000
eval_interval = 1000
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 20                   # Reduced from 50
n_embd = 128                      # Reduced from 256
n_head = 4                        # Reduced from 8
n_layer = 4                       # Reduced from 6
dropout = 0.2
quantization_levels = 512         # Reduced from 1024
mse_weight = 0.1
max_samples = 1_000_000           # Reduced from 2M
use_gradient_checkpointing = True # Enable gradient checkpointing
# -------------------------------------------------

print(f"Device: {device}")
torch.manual_seed(1337)
if device == 'cuda':
    torch.cuda.manual_seed_all(1337)

# ---------------- Load & Preprocess ----------------
def load_eeg_files(data_dir: Path, max_samples: int = None):
    """Load EDF/FIF EEG files and return flattened numpy array of raw samples (float)."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    all_signals = []
    total_loaded = 0
    print("Loading EEG files...")
    # load EDF
    for file_path in sorted(data_dir.glob("*.edf")):
        print(f"  Loading {file_path.name}")
        raw = mne.io.read_raw_edf(str(file_path), preload=True, verbose=False)
        arr = raw.get_data().flatten()
        if max_samples is not None:
            remaining = max_samples - total_loaded
            if remaining <= 0:
                break
            if len(arr) > remaining:
                arr = arr[:remaining]
        all_signals.append(arr)
        total_loaded += len(arr)
        if max_samples is not None and total_loaded >= max_samples:
            break
    # load FIF
    if max_samples is None or total_loaded < max_samples:
        for file_path in sorted(data_dir.glob("*.fif")):
            print(f"  Loading {file_path.name}")
            raw = mne.io.read_raw_fif(str(file_path), preload=True, verbose=False)
            arr = raw.get_data().flatten()
            if max_samples is not None:
                remaining = max_samples - total_loaded
                if remaining <= 0:
                    break
                if len(arr) > remaining:
                    arr = arr[:remaining]
            all_signals.append(arr)
            total_loaded += len(arr)
            if max_samples is not None and total_loaded >= max_samples:
                break
    if len(all_signals) == 0:
        raise RuntimeError(f"No EDF/FIF files found in {data_dir}")
    full_signal = np.concatenate(all_signals)
    print(f"Total raw signal length loaded: {len(full_signal):,} samples")
    return full_signal


def normalize_and_quantize(signal: np.ndarray, num_levels: int):
    """
    Normalize signal (z-score) and quantize into integer levels [0, num_levels-1].
    Returns:
      - tokens: 1D int numpy array of token ids (one per sample)
      - norm_params: dict with mean, std, num_levels
    """
    print("Normalizing signal (z-score) and quantizing...")
    mean = float(np.mean(signal))
    std = float(np.std(signal)) + 1e-8
    normed = (signal - mean) / std

    # Map normalized range to [0, 1] via a tanh-like squeeze to avoid extreme outliers dominating:
    # Use scaled arctanh? Simpler: clip to [-clip_std, clip_std], then map.
    clip_std = 5.0
    normed_clipped = np.clip(normed, -clip_std, clip_std)
    norm01 = (normed_clipped + clip_std) / (2 * clip_std)  # in [0,1]

    quantized = (norm01 * (num_levels - 1)).astype(np.int64)
    tokens = quantized  # one token per sample

    print(f"Quantized to {len(tokens):,} tokens (vocab size {num_levels})")
    norm_params = {
        'mean': mean,
        'std': std,
        'clip_std': clip_std,
        'num_levels': int(num_levels)
    }
    return tokens, norm_params


def dequantize_from_expected(expected_level, norm_params):
    """
    Convert expected_level in [0..num_levels-1] float to waveform value (original scale).
    expected_level can be torch tensor of shape (...).
    """
    num_levels = norm_params['num_levels']
    clip_std = norm_params['clip_std']
    mean = norm_params['mean']
    std = norm_params['std']

    # expected_level in [0, num_levels-1] => norm01 in [0,1]
    norm01 = expected_level / float(num_levels - 1)
    # map back to clipped normalized value in [-clip_std, clip_std]
    normed_clipped = norm01 * (2 * clip_std) - clip_std
    # map back to original scale
    waveform = normed_clipped * std + mean
    return waveform


# ---------- Set data folder ----------
filtered_dir = Path("/kaggle/input/dataset")

# Load
signal = load_eeg_files(filtered_dir, max_samples=max_samples)

# Normalize + quantize to tokens
tokens_np, norm_params = normalize_and_quantize(signal, quantization_levels)
data = torch.tensor(tokens_np, dtype=torch.long)

vocab_size = quantization_levels
print(f"Vocabulary size (single-sample): {vocab_size}")

# ---------------- Data split ----------------
chunk_size = len(data) // 5
if chunk_size == 0:
    raise RuntimeError("Not enough tokens for the split; reduce max_samples or block_size")

chunks = [data[i*chunk_size:(i+1)*chunk_size] for i in range(5)]
remainder = len(data) - chunk_size * 5
if remainder > 0:
    chunks[-1] = torch.cat([chunks[-1], data[-remainder:]])

train_data = torch.cat([chunks[0], chunks[1], chunks[3], chunks[4]])
val_data = chunks[2]

print(f"\nData split:")
print(f"Total tokens: {len(data):,}")
print(f"Train tokens: {len(train_data):,} ({100 * len(train_data) / len(data):.1f}%)")
print(f"Val tokens: {len(val_data):,} ({100 * len(val_data) / len(data):.1f}%)")

# ---------------- Batch getter ----------------
def get_batch(split: str):
    data_split = train_data if split == 'train' else val_data
    max_start = len(data_split) - block_size - 1
    if max_start <= 0:
        raise RuntimeError("Dataset too small for block_size. Decrease block_size or increase max_samples.")
    ix = torch.randint(0, max_start + 1, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# ---------------- Model components ----------------
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        tril = self.tril[:T, :T].to(wei.device)
        wei = wei.masked_fill(tril == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class EEGTransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, norm_params=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))[None, :, :]  # (1, T, C)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        mse_loss = None
        if targets is not None:
            # Cross-entropy loss
            B_, T_, C_ = logits.shape
            logits_flat = logits.view(B_ * T_, C_)
            targets_flat = targets.view(B_ * T_)
            ce_loss = F.cross_entropy(logits_flat, targets_flat)

            # MSE on waveform: compute expected quantized level from softmax probabilities
            probs = F.softmax(logits, dim=-1)  # (B, T, V)
            # create levels vector [0, 1, ..., V-1] on correct device
            levels = torch.arange(0, vocab_size, dtype=probs.dtype, device=probs.device)  # (V,)
            # expected_level = sum_k probs[..., k] * k
            expected_level = (probs @ levels)  # (B, T)
            # convert expected_level and true targets to waveform via dequantize mapping
            expected_wave = dequantize_from_expected(expected_level, norm_params)  # (B, T) float
            true_wave = dequantize_from_expected(targets.to(expected_level.dtype), norm_params)  # (B, T)
            mse_loss = F.mse_loss(expected_wave, true_wave)

            loss = ce_loss + mse_weight * mse_loss

        return logits, loss, mse_loss

    def generate(self, idx, max_new_tokens):
        """Autoregressive generation returning token IDs."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :]  # (B, vocab)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

# ---------------- Estimate loss (evaluation) ----------------
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = []
        mses = []
        for k in range(eval_iters):
            try:
                X, Y = get_batch(split)
                _, loss, mse = model(X, Y, norm_params=norm_params)
                losses.append(float(loss.item()))
                mses.append(float(mse.item()) if mse is not None else 0.0)
            except Exception:
                break
        out[split] = {
            'loss': float(np.mean(losses)) if len(losses) > 0 else 0.0,
            'mse': float(np.mean(mses)) if len(mses) > 0 else 0.0
        }
    model.train()
    return out

# --------------- Instantiate model & optimizer --------------
model = EEGTransformer(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

if device == 'cuda':
    scaler = GradScaler()
    amp_ctx = autocast
else:
    scaler = None
    amp_ctx = nullcontext

print(f"\nModel has {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
print(f"Effective batch size: {batch_size * accumulation_steps}")

# ---------------- Training loop (with accumulation) ----------------
train_losses = []
val_losses = []
train_mses = []
val_mses = []

print(f"\n{'='*60}")
print("Starting training...")
print(f"{'='*60}\n")

for step in trange(max_iters):
    # evaluation & checkpointing
    if (step + 1) % eval_interval == 0 or step == 0:
        stats = estimate_loss()
        train_losses.append(stats['train']['loss'])
        val_losses.append(stats['val']['loss'])
        train_mses.append(stats['train']['mse'])
        val_mses.append(stats['val']['mse'])
        print(f"\nStep {step}: train loss {stats['train']['loss']:.4f}, val loss {stats['val']['loss']:.4f}")
        print(f"           train mse {stats['train']['mse']:.6f}, val mse {stats['val']['mse']:.6f}")

        # save checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step,
            'vocab_size': vocab_size,
            'norm_params': norm_params,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_mses': train_mses,
            'val_mses': val_mses,
            'hyperparameters': {
                'batch_size': batch_size,
                'accumulation_steps': accumulation_steps,
                'block_size': block_size,
                'n_embd': n_embd,
                'n_head': n_head,
                'n_layer': n_layer,
                'dropout': dropout,
                'quantization_levels': quantization_levels,
                'mse_weight': mse_weight
            }
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    optimizer.zero_grad(set_to_none=True)

    # gradient accumulation
    for micro_step in range(accumulation_steps):
        xb, yb = get_batch('train')
        with amp_ctx():
            _, loss, _ = model(xb, yb, norm_params=norm_params)
            loss = loss / accumulation_steps
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

# ---------------- End training ----------------
print(f"\n{'='*60}")
print("Training completed!")
print(f"{'='*60}\n")

# ---------------- Plot losses ----------------
plt.figure(figsize=(10, 6))
num_points = len(train_losses)
eval_steps = [i * eval_interval for i in range(num_points)]
if len(eval_steps) > 0 and eval_steps[0] != 0:
    eval_steps[0] = 0
plt.plot(eval_steps, train_losses, label='Train Loss', marker='o', linewidth=2)
plt.plot(eval_steps, val_losses, label='Validation Loss', marker='s', linewidth=2)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('EEG Transformer Training & Validation Loss (CE + MSE)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('eeg_transformer_loss_ce_mse.png', dpi=150)
print("Loss plot saved to eeg_transformer_loss_ce_mse.png")
plt.show()

# ---------------- Generate & save example waveform ----------------
# Teacher-forced generation: seed with real EEG tokens (avoids cold-start collapse)
seed_len = 1024
seed = train_data[:seed_len].unsqueeze(0).to(device)  # (1, seed_len)
generated = model.generate(seed, max_new_tokens=2048)  # returns token ids

# Convert tokens to expected waveform using model's softmax expectation
with torch.no_grad():
    # compute logits over entire sequence and expected_level
    logits, _, _ = model(generated[:, :block_size], norm_params=norm_params)
    probs = F.softmax(logits, dim=-1)
    levels = torch.arange(0, vocab_size, dtype=probs.dtype, device=probs.device)
    expected_level = (probs @ levels)  # (1, T)
    waveform = dequantize_from_expected(expected_level, norm_params).squeeze(0).cpu().numpy()

# Plot a segment of the waveform
plt.figure(figsize=(12, 4))
plt.plot(waveform[:4000])
plt.title("Generated EEG waveform (teacher-forced seed)")
plt.xlabel("Sample")
plt.ylabel("Amplitude (original units)")
plt.tight_layout()
plt.savefig('generated_eeg_teacher_seed.png', dpi=150)
print("Generated waveform saved to generated_eeg_teacher_seed.png")
plt.show()

print("\n✅ Script finished. Check saved checkpoint and generated plot.")