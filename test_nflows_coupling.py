import torch
import nflows.transforms.coupling
from torch import nn

# Mock Create Net
def create_net(in_channels, out_channels):
    # nflows expects the net to take (B, in_channels, ...) and return (B, out_channels, ...)
    # For image data: (B, C, H, W)
    net = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.Conv2d(out_channels, out_channels, 1) # Initialize output to 0 for identity start
    )
    # Correct initialization
    net[-1].weight.data.zero_()
    net[-1].bias.data.zero_()
    return net

def test():
    c, h, w = 4, 8, 8
    mask = torch.ones(c)
    mask[1::2] = 0
    
    print(f"Mask shape: {mask.shape}")
    
    try:
        t = nflows.transforms.coupling.AffineCouplingTransform(
            mask=mask,
            transform_net_create_fn=create_net
        )
        print("Transform created successfully.")
    except Exception as e:
        print(f"Transform creation failed: {e}")
        return

    x = torch.randn(2, c, h, w)
    try:
        z, log_det = t(x)
        print("Forward pass successful.")
        print("Output shape:", z.shape)
        print("Log det shape:", log_det.shape)
    except Exception as e:
        print(f"Forward pass failed: {e}")

if __name__ == "__main__":
    test()