"""
Test integration of DiscretePixelCNN with Ising energy function
"""
import torch
from model import DiscretePixelCNN
from main import create_ising_energy_fn

def test_discrete_pixelcnn_init():
    """Test if DiscretePixelCNN initializes properly"""
    print("Testing DiscretePixelCNN initialization...")

    hparams = {
        "size": 8,  # Small lattice for testing
        "fix_first": 1,
        "batch_size": 4,
        "num_beta": 2,
        "beta_min": 0.1,
        "beta_max": 1.0,
        "kernel_size": 7,
        "hidden_channels": 32,
        "hidden_conv_layers": 3,
        "hidden_kernel_size": 3,
        "hidden_width": 64,
        "hidden_fc_layers": 2,
    }

    model = DiscretePixelCNN(hparams, device="cpu")
    print(f"✓ Model initialized successfully")
    print(f"  - Size: {model.size}")
    print(f"  - Fix first: {model.fix_first}")
    print(f"  - Masked conv: {type(model.masked_conv).__name__}")

    return model

def test_sampling(model):
    """Test if sampling works"""
    print("\nTesting sampling...")

    batch_size = 2
    T = torch.tensor([0.5, 1.0])

    samples = model.sample(batch_size=batch_size, T=T)
    print(f"✓ Sampling successful")
    print(f"  - Sample shape: {samples.shape}")
    print(f"  - Sample values: {samples.unique()}")
    print(f"  - First spin fixed: {samples[:, 0, 0, 0]}")

    return samples

def test_log_prob(model, samples):
    """Test if log_prob works"""
    print("\nTesting log_prob...")

    T = torch.tensor([0.5, 1.0])
    log_prob = model.log_prob(samples, T=T)
    print(f"✓ Log prob calculation successful")
    print(f"  - Log prob shape: {log_prob.shape}")
    print(f"  - Log prob values: {log_prob.squeeze()}")

    return log_prob

def test_energy_function():
    """Test if Ising energy function works"""
    print("\nTesting Ising energy function...")

    L = 8
    device = "cpu"
    energy_fn = create_ising_energy_fn(L=L, d=2, device=device)

    # Create test samples
    batch_size = 2
    samples = torch.ones(batch_size, 1, L, L)
    samples[1] = -samples[1]  # All spins down for second sample

    energy = energy_fn(samples)
    print(f"✓ Energy calculation successful")
    print(f"  - Energy shape: {energy.shape}")
    print(f"  - All spins up energy: {energy[0].item()}")
    print(f"  - All spins down energy: {energy[1].item()}")

    return energy_fn

def test_loss_calculation(model, energy_fn):
    """Test if loss calculation works (mimicking Trainer)"""
    print("\nTesting loss calculation...")

    batch_size = 4
    T = torch.tensor([0.5, 1.0, 1.5, 2.0])

    # Sample
    samples = model.sample(batch_size=batch_size, T=T)

    # Log prob
    log_prob = model.log_prob(samples, T=T)

    # Energy
    energy = energy_fn(samples)

    # Beta scaling
    beta = 1.0 / T.unsqueeze(-1)

    # Loss
    loss = log_prob + beta * energy

    print(f"✓ Loss calculation successful")
    print(f"  - Loss shape: {loss.shape}")
    print(f"  - Loss values: {loss.squeeze()}")
    print(f"  - Mean loss: {loss.mean().item()}")

def main():
    print("="*60)
    print("DiscretePixelCNN + Ising Integration Test")
    print("="*60)

    try:
        # Test model initialization
        model = test_discrete_pixelcnn_init()

        # Test sampling
        samples = test_sampling(model)

        # Test log prob
        log_prob = test_log_prob(model, samples)

        # Test energy function
        energy_fn = test_energy_function()

        # Test loss calculation
        test_loss_calculation(model, energy_fn)

        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60)

    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
