import torch
from models.motion_encoder import MotionEncoder


def main():
    # Dummy input dimensions
    B = 2    # batch size
    T = 50   # time steps
    N = 21   # joints
    d = 3    # dimensions per joint

    # Create dummy input
    x = torch.randn(B, T, N, d)

    # Initialize model
    model = MotionEncoder(
        num_joints=N,
        joint_dim=d,
        hidden_dim=128
    )

    # Forward pass
    out = model(x)

    # Print shapes
    print("Input shape :", x.shape)
    print("Output shape:", out.shape)

    # Assertions
    assert out.shape == (B, T, 128)
    print(" MotionEncoder forward pass works!")


if __name__ == "__main__":
    main()
