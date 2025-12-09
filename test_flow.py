"""
Unit and integration tests for CheckerboardFlowModel
"""
import torch
import sys
sys.path.append('.')

from model import CheckerboardFlowModel, create_checkerboard_mask


def test_checkerboard_mask():
    """Test checkerboard mask generation."""
    H, W = 4, 4
    mask0 = create_checkerboard_mask(H, W, parity=0)
    mask1 = create_checkerboard_mask(H, W, parity=1)

    # Masks should be complementary
    assert (mask0 | mask1).all(), "Mask union should cover all positions"
    assert not (mask0 & mask1).any(), "Mask intersection should be empty"

    # Each should have half the elements
    assert mask0.sum() == 8, f"Parity 0 mask should have 8 elements, got {mask0.sum()}"
    assert mask1.sum() == 8, f"Parity 1 mask should have 8 elements, got {mask1.sum()}"

    print("✓ test_checkerboard_mask passed")


def test_flow_invertibility():
    """Test that flow is invertible."""
    hparams = {
        "size": 8,
        "batch_size": 4,
        "num_beta": 2,
        "beta_min": 0.2,
        "beta_max": 1.0,
        "num_flow_layers": 4,
        "hidden_channels": 32,
        "num_hidden_layers": 1,
        "dequant_noise": 0.05,
    }

    model = CheckerboardFlowModel(hparams, device="cpu")
    model.eval()

    T = torch.tensor([[1.0], [2.0], [1.5], [0.8]])
    x = torch.randn(4, 1, 8, 8) * 0.5  # Start with small values

    with torch.no_grad():
        z, _ = model.forward_flow(x, T)
        x_recon = model.inverse_flow(z, T)

    error = (x - x_recon).abs().max().item()
    assert error < 1e-4, f"Invertibility error too large: {error}"

    print(f"✓ test_flow_invertibility passed (max error: {error:.2e})")


def test_sample_shape():
    """Test sample output shape and values."""
    hparams = {
        "size": 16,
        "batch_size": 8,
        "num_beta": 2,
        "beta_min": 0.2,
        "beta_max": 1.0,
        "num_flow_layers": 4,
        "hidden_channels": 32,
        "num_hidden_layers": 1,
        "dequant_noise": 0.05,
    }

    model = CheckerboardFlowModel(hparams, device="cpu")
    model.eval()

    T = torch.tensor([1.0, 2.0, 1.5, 0.8])

    with torch.no_grad():
        samples = model.sample(batch_size=4, T=T)

    assert samples.shape == (4, 1, 16, 16), f"Shape mismatch: {samples.shape}"

    unique_vals = set(samples.unique().tolist())
    assert unique_vals.issubset({-1.0, 1.0, 0.0}), f"Unexpected values: {unique_vals}"

    print(f"✓ test_sample_shape passed")


def test_log_prob_finite():
    """Test that log probability is finite."""
    hparams = {
        "size": 8,
        "batch_size": 4,
        "num_beta": 2,
        "beta_min": 0.2,
        "beta_max": 1.0,
        "num_flow_layers": 4,
        "hidden_channels": 32,
        "num_hidden_layers": 1,
        "dequant_noise": 0.05,
    }

    model = CheckerboardFlowModel(hparams, device="cpu")
    model.train()

    # Generate discrete samples
    samples = torch.randint(0, 2, (4, 1, 8, 8)).float() * 2 - 1
    T = torch.tensor([1.0, 2.0, 1.5, 0.8])

    log_prob = model.log_prob(samples, T=T)

    assert log_prob.shape == (4, 1), f"Shape mismatch: {log_prob.shape}"
    assert not torch.isnan(log_prob).any(), "NaN in log_prob"
    assert not torch.isinf(log_prob).any(), "Inf in log_prob"

    print(f"✓ test_log_prob_finite passed (log_prob range: [{log_prob.min():.2f}, {log_prob.max():.2f}])")


def test_training_step():
    """Test that one training step works."""
    hparams = {
        "size": 8,
        "batch_size": 4,
        "num_beta": 2,
        "beta_min": 0.2,
        "beta_max": 1.0,
        "num_flow_layers": 4,
        "hidden_channels": 32,
        "num_hidden_layers": 1,
        "dequant_noise": 0.05,
    }

    model = CheckerboardFlowModel(hparams, device="cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Simple energy function (instead of real Ising energy)
    def dummy_energy_fn(x):
        return x.view(x.shape[0], -1).sum(dim=-1, keepdim=True)

    model.train()
    T = torch.tensor([1.0, 2.0, 1.5, 0.8])

    # Continuous samples
    samples_cont = model.sample_continuous(batch_size=4, T=T)
    samples_disc = model.dequantizer.quantize(samples_cont)

    log_prob = model.log_prob(samples_cont, T=T)
    energy = dummy_energy_fn(samples_cont)
    beta = (1.0 / T).view(-1, 1)

    loss = (log_prob + beta * energy).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert not torch.isnan(loss), "NaN loss"

    print(f"✓ test_training_step passed (loss: {loss.item():.4f})")


def test_gradient_flow():
    """Test that gradients flow through the model."""
    hparams = {
        "size": 8,
        "batch_size": 4,
        "num_beta": 2,
        "beta_min": 0.2,
        "beta_max": 1.0,
        "num_flow_layers": 4,
        "hidden_channels": 32,
        "num_hidden_layers": 1,
        "dequant_noise": 0.05,
    }

    model = CheckerboardFlowModel(hparams, device="cpu")
    model.train()

    T = torch.tensor([1.0, 2.0, 1.5, 0.8])
    samples_cont = model.sample_continuous(batch_size=4, T=T)

    # Check that samples require grad (through the model parameters)
    log_prob = model.log_prob(samples_cont, T=T)
    loss = log_prob.mean()
    loss.backward()

    # Check that at least some parameters have gradients
    has_grad = False
    for param in model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break

    assert has_grad, "No gradients computed for model parameters"

    print("✓ test_gradient_flow passed")


if __name__ == "__main__":
    print("Running CheckerboardFlowModel tests...\n")

    test_checkerboard_mask()
    test_flow_invertibility()
    test_sample_shape()
    test_log_prob_finite()
    test_training_step()
    test_gradient_flow()

    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
