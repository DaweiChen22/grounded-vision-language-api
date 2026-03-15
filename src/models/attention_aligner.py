import torch
import torch.nn as nn

class CrossModalAttention(nn.Module):
    """Transformer-based cross-attention for aligning visual and textual tokens."""
    def __init__(self, d_model=512, nhead=8):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead)

    def forward(self, visual_tokens, textual_tokens):
        """Aligns visual perceptual data with linguistic intent."""
        # Q = Text, K,V = Vision
        attn_output, attn_weights = self.multihead_attn(textual_tokens, visual_tokens, visual_tokens)
        return attn_output, attn_weights

print("Cross-Modal Perceptual Aligner Loaded.")
