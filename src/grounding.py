import torch
import torch.nn as nn

class GroundedEngine(nn.Module):
    """Bridges perceptual visual data with linguistic tokens."""
    def __init__(self):
        super().__init__()
        self.vision_encoder = nn.Linear(2048, 512)
        self.text_encoder = nn.Linear(1024, 512)

    def forward(self, img_feat, txt_feat):
        v = self.vision_encoder(img_feat)
        t = self.text_encoder(txt_feat)
        return torch.cosine_similarity(v, t)

print("Multimodal Grounding Module Initialized.")
