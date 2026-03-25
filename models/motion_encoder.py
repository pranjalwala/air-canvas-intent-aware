import torch
import torch.nn as nn


class MotionEncoder(nn.Module):
    """
    Encodes mid-air hand trajectories as temporal feature sequences.

    Input:
        x : Tensor of shape (B, T, N, d)
    Output:
        h : Tensor of shape (B, T, D)
    """

    def __init__(self, num_joints=21, joint_dim=3, hidden_dim=128):
        super().__init__()

        self.num_joints = num_joints
        self.joint_dim = joint_dim
        self.hidden_dim = hidden_dim

        # Project joint coordinates into feature space
        self.input_proj = nn.Linear(num_joints * joint_dim, hidden_dim)

        # Temporal sequence model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x (Tensor): shape (B, T, N, d)

        Returns:
            h (Tensor): shape (B, T, hidden_dim)
        """
        B, T, N, d = x.shape

        assert N == self.num_joints, "Unexpected number of joints"
        assert d == self.joint_dim, "Unexpected joint dimension"

        # Flatten joints
        x = x.view(B, T, N * d)     # (B, T, N*d)

        # Linear projection
        x = self.input_proj(x)     # (B, T, D)

        # Temporal encoding
        h = self.transformer(x)    # (B, T, D)

        return h
