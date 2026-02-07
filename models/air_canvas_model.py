import torch
import torch.nn as nn

from models.motion_encoder import MotionEncoder
from models.intent_head import IntentHead
from models.text_head import TextHead
from models.drawing_head import DrawingHead


class AirCanvasModel(nn.Module):
    """
    Unified intent-aware air-canvas model.
    Routes motion to text or drawing heads based on intent.
    """

    def __init__(
        self,
        num_joints=21,
        joint_dim=3,
        hidden_dim=128,
        vocab_size=30,
        embed_dim=256,
    ):
        super().__init__()

        self.encoder = MotionEncoder(
            num_joints=num_joints,
            joint_dim=joint_dim,
            hidden_dim=hidden_dim,
        )

        self.intent_head = IntentHead(hidden_dim=hidden_dim)
        self.text_head = TextHead(hidden_dim=hidden_dim, vocab_size=vocab_size)
        self.drawing_head = DrawingHead(hidden_dim=hidden_dim, embed_dim=embed_dim)

    def forward(self, x):
        """
        Args:
            x (Tensor): (B, T, N, d)

        Returns:
            dict with:
              - h               : (B, T, D)
              - intent_logits   : (B, 1)
              - text_log_probs  : (T, B, V)
              - drawing_emb     : (B, E)
        """
        h = self.encoder(x)                    # (B, T, D)

        intent_logits = self.intent_head(h)    # (B, 1)

        text_log_probs = self.text_head(h)     # (T, B, V)
        drawing_emb = self.drawing_head(h)     # (B, E)

        return {
            "h": h,
            "intent_logits": intent_logits,
            "text_log_probs": text_log_probs,
            "drawing_emb": drawing_emb,
        }
