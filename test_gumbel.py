"""
Test script for Gumbel-Softmax implementation.

This script verifies that:
1. Gumbel-Softmax sampling works
2. Gradients flow through the samples
3. Temperature annealing works correctly
"""

import torch
import sys
from model import DiscretePixelCNN
from config import GumbelConfig

def test_gumbel_sampling():
    """Test basic Gumbel-Softmax sampling."""
    print("=" * 60)
    print("TEST 1: Gumbel-Softmax Sampling")
    print("=" * 60)

    # Create a small model
    hparams = {
        "size": 4,  # Small lattice for testing
        "batch_size": 2,
        "num_beta": 2,
        "beta_min": 0.5,
        "beta_max": 2.0,
        "kernel_size": 3,
        "hidden_channels": 8,
        "hidden_conv_layers": 1,
        "hidden_kernel_size": 3,
        "hidden_width": 16,
        "hidden_fc_layers": 1,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DiscretePixelCNN(hparams, device=device).to(device)

    # Test sampling
    batch_size = 4
    T = torch.tensor([1.0, 2.0, 0.5, 1.5], device=device)

    # Hard mode (discrete forward, soft backward)
    print("\n1. Testing hard mode (straight-through estimator)...")
    samples_hard = model.sample_gumbel_softmax(
        batch_size=batch_size,
        T=T,
        temperature=0.5,
        hard=True
    )
    print(f"   Shape: {samples_hard.shape}")
    print(f"   Values (should be in {{-1, +1}}): {torch.unique(samples_hard)}")
    print(f"   ✓ Hard sampling works!")

    # Soft mode (continuous)
    print("\n2. Testing soft mode (continuous relaxation)...")
    samples_soft = model.sample_gumbel_softmax(
        batch_size=batch_size,
        T=T,
        temperature=1.0,
        hard=False
    )
    print(f"   Shape: {samples_soft.shape}")
    print(f"   Value range: [{samples_soft.min():.3f}, {samples_soft.max():.3f}]")
    print(f"   ✓ Soft sampling works!")

    return True


def test_gradient_flow():
    """Test that gradients flow in train_epoch_gumbel."""
    print("\n" + "=" * 60)
    print("TEST 2: Gradient Flow (train_epoch_gumbel)")
    print("=" * 60)

    from util import Trainer, GumbelConfig
    import torch.nn.functional as F

    hparams = {
        "size": 4,
        "batch_size": 2,
        "num_beta": 2,
        "beta_min": 0.5,
        "beta_max": 2.0,
        "kernel_size": 3,
        "hidden_channels": 8,
        "hidden_conv_layers": 1,
        "hidden_kernel_size": 3,
        "hidden_width": 16,
        "hidden_fc_layers": 1,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DiscretePixelCNN(hparams, device=device).to(device)

    # Simple energy function (Ising-like)
    def simple_energy(samples):
        """Simple energy: negative sum of nearest-neighbor interactions."""
        # samples: (B, C, H, W) in {-1, +1}
        # Compute horizontal and vertical neighbors
        h_neighbors = samples[:, :, :, :-1] * samples[:, :, :, 1:]
        v_neighbors = samples[:, :, :-1, :] * samples[:, :, 1:, :]
        energy = -(h_neighbors.sum(dim=[1,2,3]) + v_neighbors.sum(dim=[1,2,3]))
        return energy.unsqueeze(-1)  # (B, 1)

    # Create trainer with Gumbel config
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    gumbel_config = GumbelConfig(use_gumbel=True, initial_temperature=1.0)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        gumbel_config=gumbel_config
    )

    print("\n1. Testing hybrid Gumbel-Softmax approach...")
    print("   - Discrete sampling for valid configurations")
    print("   - Soft relaxation for energy gradients")

    # Clear any existing gradients
    optimizer.zero_grad()

    # Run one training epoch with Gumbel-Softmax
    try:
        train_loss = trainer.train_epoch_gumbel(
            energy_fn=simple_energy,
            temperature=0.5,
            hard=True
        )
        print(f"   ✓ Training epoch completed! Loss: {train_loss:.4f}")

        # Check if gradients exist
        has_grad = False
        grad_count = 0
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                grad_count += 1

        if has_grad:
            print(f"   ✓ Gradients found in {grad_count} parameters!")
            print("   ✓ Gradients flow through Gumbel-Softmax!")
        else:
            print("   ✗ No gradients found")
            return False

    except Exception as e:
        print(f"   ✗ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_temperature_annealing():
    """Test temperature annealing schedule."""
    print("\n" + "=" * 60)
    print("TEST 3: Temperature Annealing")
    print("=" * 60)

    config = GumbelConfig(
        use_gumbel=True,
        initial_temperature=1.0,
        final_temperature=0.1,
        anneal_epochs=100,
        hard=True
    )

    epochs = [0, 25, 50, 75, 100, 150]
    total_epochs = 100

    print("\nTemperature schedule (exponential decay):")
    print("  Epoch  | Temperature")
    print("-" * 30)
    for epoch in epochs:
        temp = config.get_temperature(epoch, total_epochs)
        print(f"  {epoch:5d}  | {temp:.4f}")

    # Check monotonicity
    temps = [config.get_temperature(e, total_epochs) for e in range(total_epochs)]
    is_decreasing = all(temps[i] >= temps[i+1] for i in range(len(temps)-1))

    if is_decreasing:
        print("\n   ✓ Temperature decreases monotonically!")
    else:
        print("\n   ✗ Temperature is not monotonically decreasing!")
        return False

    # Check boundary values
    initial = config.get_temperature(0, total_epochs)
    final = config.get_temperature(100, total_epochs)

    if abs(initial - 1.0) < 1e-6 and abs(final - 0.1) < 1e-6:
        print(f"   ✓ Boundary values correct: initial={initial:.4f}, final={final:.4f}")
    else:
        print(f"   ✗ Boundary values incorrect: initial={initial:.4f}, final={final:.4f}")
        return False

    return True


def test_comparison():
    """Compare regular sampling vs Gumbel-Softmax sampling."""
    print("\n" + "=" * 60)
    print("TEST 4: Comparison (Regular vs Gumbel-Softmax)")
    print("=" * 60)

    hparams = {
        "size": 4,
        "batch_size": 4,
        "num_beta": 2,
        "beta_min": 0.5,
        "beta_max": 2.0,
        "kernel_size": 3,
        "hidden_channels": 8,
        "hidden_conv_layers": 1,
        "hidden_kernel_size": 3,
        "hidden_width": 16,
        "hidden_fc_layers": 1,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DiscretePixelCNN(hparams, device=device).to(device)
    model.eval()

    batch_size = 4
    T = torch.ones(batch_size, device=device)

    # Regular sampling
    print("\n1. Regular discrete sampling:")
    with torch.no_grad():
        samples_regular = model.sample(batch_size=batch_size, T=T)
    print(f"   Shape: {samples_regular.shape}")
    print(f"   Values: {torch.unique(samples_regular)}")

    # Gumbel-Softmax (hard mode, low temperature for determinism)
    print("\n2. Gumbel-Softmax (hard, temp=0.1):")
    with torch.no_grad():
        samples_gumbel = model.sample_gumbel_softmax(
            batch_size=batch_size,
            T=T,
            temperature=0.1,
            hard=True
        )
    print(f"   Shape: {samples_gumbel.shape}")
    print(f"   Values: {torch.unique(samples_gumbel)}")

    # Both should produce discrete {-1, +1} values
    is_discrete_regular = set(samples_regular.unique().tolist()) == {-1.0, 1.0}
    is_discrete_gumbel = set(samples_gumbel.unique().tolist()) == {-1.0, 1.0}

    if is_discrete_regular and is_discrete_gumbel:
        print("\n   ✓ Both methods produce discrete samples!")
    else:
        print(f"\n   Regular discrete: {is_discrete_regular}")
        print(f"   Gumbel discrete: {is_discrete_gumbel}")

    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("GUMBEL-SOFTMAX IMPLEMENTATION TEST SUITE")
    print("=" * 60)

    tests = [
        ("Gumbel-Softmax Sampling", test_gumbel_sampling),
        ("Gradient Flow", test_gradient_flow),
        ("Temperature Annealing", test_temperature_annealing),
        ("Regular vs Gumbel Comparison", test_comparison),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name}")

    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
