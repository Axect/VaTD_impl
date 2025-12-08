"""
Test integration of RealNVP with Ising energy function
"""

import torch
import torch.nn.functional as F
from model import RealNVP
from main import create_ising_energy_fn
import numpy as np

def test_realnvp_init():
    """Test if RealNVP initializes properly"""
    print("Testing RealNVP initialization...")

    hparams = {
        "size": 8,  # Small lattice for testing
        "batch_size": 4,
        "num_beta": 2,
        "beta_min": 0.1,
        "beta_max": 1.0,
        "hidden_channels": 32,
        "hidden_conv_layers": 2,
        "num_flow_layers": 2,
    }

    model = RealNVP(hparams, device="cpu")
    print(f"✓ Model initialized successfully")
    print(f"  - Size: {model.size}")
    print(f"  - In channels: {model.in_channels}")
    
    return model


def test_sampling(model):
    """Test if sampling works"""
    print("\nTesting sampling...")

    batch_size = 4
    T = torch.tensor([0.5, 1.0, 1.5, 2.0])

    # Sample
    samples = model.sample(batch_size=batch_size, T=T)
    print(f"✓ Sampling successful")
    print(f"  - Sample shape: {samples.shape}")
    print(f"  - Min value: {samples.min().item()}")
    print(f"  - Max value: {samples.max().item()}")
    
    assert samples.min() >= -1.0, "Samples should be >= -1"
    assert samples.max() <= 1.0, "Samples should be <= 1"

    return samples


def test_log_prob(model, samples):
    """Test if log_prob works"""
    print("\nTesting log_prob...")

    batch_size = samples.shape[0]
    T = torch.tensor([0.5, 1.0, 1.5, 2.0])[:batch_size]
    
    log_prob = model.log_prob(samples, T=T)
    print(f"✓ Log prob calculation successful")
    print(f"  - Log prob shape: {log_prob.shape}")
    print(f"  - Log prob values: {log_prob.squeeze()}")

    # Check for NaNs
    assert not torch.isnan(log_prob).any(), "Log prob contains NaNs"
    
    # Test boundary conditions (near -1 and 1) to ensure stability
    # Construct artificial samples near boundaries
    boundary_samples = samples.clone()
    boundary_samples = torch.clamp(boundary_samples, -0.999, 0.999)
    # Push some values very close to 1
    boundary_samples[0, 0, 0, 0] = 0.999999
    
    log_prob_boundary = model.log_prob(boundary_samples, T=T)
    print(f"  - Log prob (boundary): {log_prob_boundary.squeeze()}")
    assert not torch.isnan(log_prob_boundary).any(), "Log prob (boundary) contains NaNs"

    return log_prob


def main():
    print("=" * 60)
    print("RealNVP Integration Test")
    print("=" * 60)

    try:
        # Test model initialization
        model = test_realnvp_init()

        # Test sampling
        samples = test_sampling(model)

        # Test log prob
        test_log_prob(model, samples)

        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
