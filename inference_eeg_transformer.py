import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

"""
inference_eeg_transformer.py

Load trained transformer and generate EEG signals.
Decodes tokens back to signal values and plots them.
"""

# ----------------- Model Architecture (must match training) -------------------
class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size, dropout):
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
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
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
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class EEGTransformer(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
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
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))[None, :, :]
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        mse_loss = None
        if targets is not None:
            B_, T_, C_ = logits.shape
            logits_flat = logits.view(B_ * T_, C_)
            targets_flat = targets.view(B_ * T_)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss, mse_loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens autoregressively.
        
        Args:
            idx: (B, T) initial context
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature (higher = more random)
            top_k: if set, only sample from top k logits
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ----------------- Token to Signal Conversion -------------------
def tokens_to_signal(tokens, norm_params):
    """
    Convert tokens back to signal values using the same dequantization as training.
    Each token is a single quantized sample (not a pair).
    """
    num_levels = norm_params['num_levels']
    clip_std = norm_params['clip_std']
    mean = norm_params['mean']
    std = norm_params['std']
    
    # tokens are quantized levels in [0, num_levels-1]
    # map back to [0,1]
    norm01 = tokens / (num_levels - 1)
    
    # map back to clipped normalized value in [-clip_std, clip_std]
    normed_clipped = norm01 * (2 * clip_std) - clip_std
    
    # map back to original scale (z-score inverse)
    signal = normed_clipped * std + mean
    
    return signal


# ----------------- Load Model and Generate -------------------
def load_model_and_generate(checkpoint_path, num_tokens=500, temperature=0.8, top_k=100, seed_length=20):
    """
    Load trained model and generate EEG signal.
    
    Args:
        checkpoint_path: path to saved checkpoint
        num_tokens: number of tokens to generate
        temperature: sampling temperature
        top_k: top-k sampling parameter
        seed_length: length of random seed context
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract hyperparameters
    hyperparams = checkpoint['hyperparameters']
    vocab_size = checkpoint['vocab_size']
    norm_params = checkpoint['norm_params']
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Hyperparameters: {hyperparams}")
    
    # Instantiate model
    model = EEGTransformer(
        vocab_size=vocab_size,
        n_embd=hyperparams['n_embd'],
        n_head=hyperparams['n_head'],
        n_layer=hyperparams['n_layer'],
        block_size=hyperparams['block_size'],
        dropout=hyperparams['dropout']
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"\nModel loaded successfully!")
    print(f"Model has {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # Generate signal
    print(f"\nGenerating {num_tokens} tokens...")
    print(f"Temperature: {temperature}, Top-k: {top_k}")
    
    # Create random seed context
    torch.manual_seed(42)
    context = torch.randint(0, vocab_size, (1, seed_length), dtype=torch.long, device=device)
    
    with torch.no_grad():
        generated = model.generate(context, max_new_tokens=num_tokens, temperature=temperature, top_k=top_k)
    
    # Convert to numpy
    generated_tokens = generated[0].cpu().numpy()
    print(f"Generated {len(generated_tokens)} tokens total (including seed)")
    
    # Convert tokens to signal
    print("\nConverting tokens to signal...")
    signal = tokens_to_signal(generated_tokens, norm_params)
    print(f"Signal length: {len(signal)} samples")
    
    return signal, generated_tokens, norm_params


# ----------------- Plotting -------------------
def plot_generated_signal(signal, output_path='generated_eeg_signal.png', num_samples_to_plot=2000):
    """Plot the generated EEG signal."""
    print(f"\nPlotting signal (first {num_samples_to_plot} samples)...")
    
    # Limit samples for visualization
    signal_plot = signal[:num_samples_to_plot]
    time = np.arange(len(signal_plot))
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Full view
    axes[0].plot(time, signal_plot, linewidth=0.7, color='#2E86AB')
    axes[0].set_xlabel('Sample', fontsize=12)
    axes[0].set_ylabel('Amplitude (μV)', fontsize=12)
    axes[0].set_title('Generated EEG Signal (Full View)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Zoomed view (first 500 samples)
    zoom_samples = min(500, len(signal_plot))
    axes[1].plot(time[:zoom_samples], signal_plot[:zoom_samples], linewidth=1.2, color='#A23B72')
    axes[1].set_xlabel('Sample', fontsize=12)
    axes[1].set_ylabel('Amplitude (μV)', fontsize=12)
    axes[1].set_title(f'Generated EEG Signal (Zoomed: First {zoom_samples} samples)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved to {output_path}")
    plt.show()


def plot_signal_statistics(signal, output_path='generated_eeg_statistics.png'):
    """Plot statistics of the generated signal."""
    print("\nPlotting signal statistics...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Histogram
    axes[0, 0].hist(signal, bins=50, color='#2E86AB', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Amplitude (μV)', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Amplitude Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Power spectrum
    fft = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal))
    power = np.abs(fft)**2
    
    # Only plot positive frequencies
    positive_freqs = freqs[:len(freqs)//2]
    positive_power = power[:len(power)//2]
    
    axes[0, 1].plot(positive_freqs * 400, positive_power, linewidth=1, color='#A23B72')  # assuming 400 Hz
    axes[0, 1].set_xlabel('Frequency (Hz)', fontsize=11)
    axes[0, 1].set_ylabel('Power', fontsize=11)
    axes[0, 1].set_title('Power Spectrum', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlim(0, 60)  # Focus on 0-60 Hz
    axes[0, 1].grid(True, alpha=0.3)
    
    # Autocorrelation
    autocorr = np.correlate(signal - np.mean(signal), signal - np.mean(signal), mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / autocorr[0]
    lags = np.arange(min(500, len(autocorr)))
    
    axes[1, 0].plot(lags, autocorr[:len(lags)], linewidth=1.5, color='#F18F01')
    axes[1, 0].set_xlabel('Lag (samples)', fontsize=11)
    axes[1, 0].set_ylabel('Autocorrelation', fontsize=11)
    axes[1, 0].set_title('Autocorrelation', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Statistics text
    stats_text = f"""
    Mean: {np.mean(signal):.4f} μV
    Std: {np.std(signal):.4f} μV
    Min: {np.min(signal):.4f} μV
    Max: {np.max(signal):.4f} μV
    Length: {len(signal)} samples
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Signal Statistics', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Statistics plot saved to {output_path}")
    plt.show()


# ----------------- Main -------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate EEG signals from trained transformer')
    parser.add_argument('--checkpoint', type=str, default='eeg_transformer.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--num_tokens', type=int, default=1000,
                        help='Number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.2,
                        help='Sampling temperature (higher = more random)')
    parser.add_argument('--top_k', type=int, default=1000,
                        help='Top-k sampling parameter')
    parser.add_argument('--seed_length', type=int, default=20,
                        help='Length of seed context')
    parser.add_argument('--samples_to_plot', type=int, default=2000,
                        help='Number of samples to plot')
    
    args = parser.parse_args()
    
    print(f"{'='*60}")
    print("EEG Transformer - Signal Generation")
    print(f"{'='*60}\n")
    
    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return
    
    # Generate signal
    signal, tokens, norm_params = load_model_and_generate(
        checkpoint_path=args.checkpoint,
        num_tokens=args.num_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed_length=args.seed_length
    )
    
    # Plot
    plot_generated_signal(signal, num_samples_to_plot=args.samples_to_plot)
    plot_signal_statistics(signal)
    
    print(f"\n{'='*60}")
    print("✅ Generation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
