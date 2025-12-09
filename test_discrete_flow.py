"""
Test script for DiscreteFlowModel implementation.

Tests:
1. BinaryCouplingLayer self-inverse property
2. DiscreteFlowModel sample range
3. DiscreteFlowModel log_prob computation
4. Gradient flow through STE
"""

import torch
from model import BinaryCouplingLayer, DiscreteFlowModel, ste_round

def test_ste_round():
    """Test Straight-Through Estimator."""
    print("Testing STE Round...")

    x = torch.tensor([0.2, 0.6, 0.8], requires_grad=True)
    y = ste_round(x)

    # Forward: should round (note: torch.round uses banker's rounding, 0.5 -> 0)
    expected = torch.tensor([0.0, 1.0, 1.0])
    assert torch.allclose(y, expected), f"Expected {expected}, got {y}"

    # Backward: gradient should pass through
    loss = y.sum()
    loss.backward()

    # Gradient should be all ones (identity)
    expected_grad = torch.ones(3)
    assert torch.allclose(x.grad, expected_grad), f"Expected grad {expected_grad}, got {x.grad}"

    print("✓ STE Round test passed")


def test_binary_coupling_self_inverse():
    """Test BinaryCouplingLayer self-inverse property."""
    print("\nTesting BinaryCouplingLayer self-inverse...")

    layer = BinaryCouplingLayer(H=4, W=4, parity=0)
    x = torch.randint(0, 2, (8, 1, 4, 4)).float()
    T = torch.ones(8, 1) * 2.0

    # Forward
    z, log_det = layer.forward(x, T)

    # Inverse (should recover x)
    x_recon = layer.inverse(z, T)

    # Check if x == x_recon
    assert torch.allclose(x, x_recon, atol=1e-5), "Self-inverse property violated!"

    # Check log_det is 0
    assert torch.allclose(log_det, torch.zeros_like(log_det)), "Log det should be 0!"

    print(f"✓ Self-inverse test passed (max diff: {(x - x_recon).abs().max().item():.2e})")


def test_discrete_flow_sample_range():
    """Test DiscreteFlowModel samples are in {-1, +1}."""
    print("\nTesting DiscreteFlowModel sample range...")

    hparams = {
        "size": 4,
        "batch_size": 16,
        "num_beta": 2,
        "beta_min": 0.2,
        "beta_max": 1.0,
        "num_flow_layers": 3,
        "hidden_channels": 16,
        "num_hidden_layers": 1,
    }

    model = DiscreteFlowModel(hparams, device="cpu")

    # Sample
    T = torch.ones(16, 1) * 2.0
    samples = model.sample(batch_size=16, T=T)

    # Check shape
    assert samples.shape == (16, 1, 4, 4), f"Wrong shape: {samples.shape}"

    # Check range: should be exactly {-1, +1}
    unique_vals = torch.unique(samples)
    print(f"  Unique values in samples: {unique_vals.tolist()}")

    assert torch.all((samples == -1) | (samples == 1)), "Samples not in {-1, +1}!"

    print("✓ Sample range test passed")


def test_discrete_flow_log_prob():
    """Test DiscreteFlowModel log_prob computation."""
    print("\nTesting DiscreteFlowModel log_prob...")

    hparams = {
        "size": 4,
        "batch_size": 16,
        "num_beta": 2,
        "beta_min": 0.2,
        "beta_max": 1.0,
        "num_flow_layers": 3,
        "hidden_channels": 16,
        "num_hidden_layers": 1,
    }

    model = DiscreteFlowModel(hparams, device="cpu")

    # Create random samples in {-1, +1}
    samples = torch.randint(0, 2, (8, 1, 4, 4)).float() * 2 - 1
    T = torch.ones(8, 1) * 2.0

    # Compute log prob
    log_prob = model.log_prob(samples, T=T)

    # Check shape
    assert log_prob.shape == (8, 1), f"Wrong shape: {log_prob.shape}"

    # Check all finite
    assert torch.all(torch.isfinite(log_prob)), "Non-finite log_prob detected!"

    print(f"  Log prob range: [{log_prob.min().item():.2f}, {log_prob.max().item():.2f}]")
    print("✓ Log prob test passed")


def test_gradient_flow():
    """Test gradient flow through STE."""
    print("\nTesting gradient flow through STE...")

    hparams = {
        "size": 4,
        "batch_size": 4,
        "num_beta": 2,
        "beta_min": 0.2,
        "beta_max": 1.0,
        "num_flow_layers": 2,
        "hidden_channels": 8,
        "num_hidden_layers": 1,
    }

    model = DiscreteFlowModel(hparams, device="cpu")

    # Sample with gradient tracking
    T = torch.ones(4, 1) * 2.0
    samples = model.sample(batch_size=4, T=T)

    # Compute log prob
    log_prob = model.log_prob(samples, T=T)

    # Simple loss
    loss = log_prob.mean()

    # Backward
    loss.backward()

    # Check if gradients exist
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().max() > 0:
            has_grad = True
            print(f"  {name}: grad norm = {param.grad.norm().item():.2e}")

    assert has_grad, "No gradients detected!"

    print("✓ Gradient flow test passed")


def test_forward_inverse_consistency():
    """Test forward-inverse consistency of full model."""
    print("\nTesting forward-inverse consistency...")

    hparams = {
        "size": 4,
        "batch_size": 8,
        "num_beta": 2,
        "beta_min": 0.2,
        "beta_max": 1.0,
        "num_flow_layers": 3,
        "hidden_channels": 16,
        "num_hidden_layers": 1,
    }

    model = DiscreteFlowModel(hparams, device="cpu")
    model.eval()

    with torch.no_grad():
        # Create binary input
        x_bin = torch.randint(0, 2, (8, 1, 4, 4)).float()
        T = torch.ones(8, 1) * 2.0

        # Forward flow
        z, _ = model.forward_flow(x_bin, T)

        # Inverse flow
        x_recon = model.inverse_flow(z, T)

        # Check reconstruction
        diff = (x_bin - x_recon).abs().max().item()
        print(f"  Max reconstruction error: {diff:.2e}")

        assert torch.allclose(x_bin, x_recon, atol=1e-5), f"Reconstruction error too large: {diff}"

    print("✓ Forward-inverse consistency test passed")


if __name__ == "__main__":
    print("="*60)
    print("DiscreteFlowModel Test Suite")
    print("="*60)

    try:
        test_ste_round()
        test_binary_coupling_self_inverse()
        test_discrete_flow_sample_range()
        test_discrete_flow_log_prob()
        test_gradient_flow()
        test_forward_inverse_consistency()

        print("\n" + "="*60)
        print("✅ All tests passed!")
        print("="*60)

    except Exception as e:
        print("\n" + "="*60)
        print(f"❌ Test failed: {e}")
        print("="*60)
        raise
