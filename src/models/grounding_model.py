import torch
import torch.nn as nn
import torch.nn.functional as F

class PerceptualAligner(nn.Module):
    """Multimodal architecture for grounding linguistic tokens in visual features."""
    def __init__(self, vision_dim=2048, text_dim=768, projection_dim=512):
        super().__init__()
        self.visual_proj = nn.Linear(vision_dim, projection_dim)
        self.text_proj = nn.Linear(text_dim, projection_dim)
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)

    def compute_grounding_score(self, image_features, text_features):
        """Calculates the alignment between visual and textual latent spaces."""
        v_emb = F.normalize(self.visual_proj(image_features), dim=-1)
        t_emb = F.normalize(self.text_proj(text_features), dim=-1)
        return torch.matmul(v_emb, t_emb.t()) * self.temperature.exp()

if __name__ == "__main__":
    model = PerceptualAligner()
    print("Grounded vision-language model architecture ready.")
