import torch

from models.losses import motion_smoothness_loss


def main():
    B, T, N, d = 2, 10, 21, 3

    # Case 1: perfectly linear motion → zero acceleration
    t = torch.linspace(0, 1, T)
    linear_motion = t.view(1, T, 1, 1).repeat(B, 1, N, d)

    loss_linear = motion_smoothness_loss(linear_motion)
    print("Linear motion loss :", loss_linear.item())

    assert torch.allclose(loss_linear, torch.tensor(0.0), atol=1e-6)

    # Case 2: random motion → non-zero acceleration
    random_motion = torch.randn(B, T, N, d)
    loss_random = motion_smoothness_loss(random_motion)

    print("Random motion loss :", loss_random.item())
    assert loss_random.item() > 0

    print(" Motion smoothness loss works correctly!")


if __name__ == "__main__":
    main()
