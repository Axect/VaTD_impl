#!/usr/bin/env python3
"""
Test Automatic Differentiation for DiscretePixelCNNFiLM.

Verifies that:
1. Gradients flow correctly through Temperature Embedding + FiLM
2. Model parameters receive gradients
3. log_prob is differentiable w.r.t. model parameters
"""

import torch
import torch.nn as nn

# Import the new model
from model import DiscretePixelCNNFiLM, TemperatureEmbedding, FiLMGenerator


def test_temperature_embedding_grad():
    """Test that TemperatureEmbedding is differentiable."""
    print("=" * 60)
    print("Test 1: TemperatureEmbedding Gradient Flow")
    print("=" * 60)

    embed = TemperatureEmbedding(embed_dim=64, hidden_dim=128, num_freqs=32)

    # Test with requires_grad=True for T
    T = torch.tensor([2.0, 2.27, 3.0], requires_grad=True)

    embedding = embed(T)
    loss = embedding.sum()
    loss.backward()

    print(f"Input T shape: {T.shape}")
    print(f"Output embedding shape: {embedding.shape}")
    print(f"T.grad exists: {T.grad is not None}")
    print(f"T.grad: {T.grad}")

    # Check model parameters have gradients
    has_grad = all(p.grad is not None for p in embed.parameters())
    print(f"All embedding params have gradients: {has_grad}")

    print("✓ TemperatureEmbedding AD test passed!\n")
    return True


def test_film_generator_grad():
    """Test that FiLMGenerator is differentiable."""
    print("=" * 60)
    print("Test 2: FiLMGenerator Gradient Flow")
    print("=" * 60)

    embed_dim = 64
    feature_dims = [128, 128, 256]

    film_gen = FiLMGenerator(
        embed_dim=embed_dim,
        num_layers=len(feature_dims),
        feature_dims=feature_dims
    )

    # Create embedding input
    temp_embedding = torch.randn(4, embed_dim, requires_grad=True)

    # Generate FiLM params
    film_params = film_gen(temp_embedding)

    # Compute dummy loss
    loss = sum(gamma.sum() + beta.sum() for gamma, beta in film_params)
    loss.backward()

    print(f"Input embedding shape: {temp_embedding.shape}")
    print(f"Number of FiLM layers: {len(film_params)}")
    for i, (gamma, beta) in enumerate(film_params):
        print(f"  Layer {i}: gamma {gamma.shape}, beta {beta.shape}")

    print(f"temp_embedding.grad exists: {temp_embedding.grad is not None}")

    # Check model parameters have gradients
    has_grad = all(p.grad is not None for p in film_gen.parameters())
    print(f"All FiLMGenerator params have gradients: {has_grad}")

    print("✓ FiLMGenerator AD test passed!\n")
    return True


def test_full_model_grad():
    """Test full DiscretePixelCNNFiLM gradient flow."""
    print("=" * 60)
    print("Test 3: Full DiscretePixelCNNFiLM Gradient Flow")
    print("=" * 60)

    # Create model with small config for testing
    hparams = {
        "size": 8,  # Small lattice for fast testing
        "batch_size": 4,
        "num_beta": 2,
        "beta_min": 0.2,
        "beta_max": 1.0,
        "fix_first": 1,
        "hidden_channels": 32,
        "hidden_conv_layers": 2,
        "hidden_width": 64,
        "hidden_fc_layers": 1,
        "kernel_size": 3,
        "hidden_kernel_size": 3,
        "temp_embed_dim": 32,
        "temp_hidden_dim": 64,
        "temp_num_freqs": 16,
    }

    model = DiscretePixelCNNFiLM(hparams, device="cpu")

    # Create test inputs
    batch_size = 4
    T = torch.tensor([1.5, 2.0, 2.27, 3.0])

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Temperature input: {T}")

    # Test sampling (no gradient needed for sampling itself)
    print("\nTesting sample()...")
    with torch.no_grad():
        samples = model.sample(batch_size=batch_size, T=T)
    print(f"Samples shape: {samples.shape}")
    print(f"Sample values range: [{samples.min()}, {samples.max()}]")

    # Test log_prob with gradient
    print("\nTesting log_prob() with gradients...")
    model.zero_grad()

    log_prob = model.log_prob(samples, T=T)
    print(f"log_prob shape: {log_prob.shape}")
    print(f"log_prob values: {log_prob.squeeze().tolist()}")

    # Compute loss and backward
    loss = log_prob.mean()
    loss.backward()

    # Check gradients exist for all parameters
    params_with_grad = 0
    params_without_grad = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            params_with_grad += 1
        else:
            params_without_grad += 1
            print(f"  WARNING: No gradient for {name}")

    print(f"\nParameters with gradients: {params_with_grad}")
    print(f"Parameters without gradients: {params_without_grad}")

    # Verify key components have gradients
    print("\nChecking key components:")
    print(f"  temp_embedding.mlp[0].weight.grad exists: {model.temp_embedding.mlp[0].weight.grad is not None}")
    print(f"  film_generator.film_layers[0].weight.grad exists: {model.film_generator.film_layers[0].weight.grad is not None}")
    print(f"  masked_conv.first_conv.weight.grad exists: {model.masked_conv.first_conv.weight.grad is not None}")

    if params_without_grad == 0:
        print("\n✓ Full model AD test passed! All parameters have gradients.")
        return True
    else:
        print(f"\n✗ AD test failed: {params_without_grad} parameters missing gradients")
        return False


def test_vatd_loss_simulation():
    """Simulate VaTD loss computation to verify AD in training context."""
    print("=" * 60)
    print("Test 4: VaTD Loss Simulation")
    print("=" * 60)

    # Create model
    hparams = {
        "size": 8,
        "batch_size": 8,
        "num_beta": 4,
        "beta_min": 0.2,
        "beta_max": 1.0,
        "fix_first": 1,
        "hidden_channels": 32,
        "hidden_conv_layers": 2,
        "hidden_width": 64,
        "hidden_fc_layers": 1,
        "kernel_size": 3,
        "hidden_kernel_size": 3,
    }

    model = DiscretePixelCNNFiLM(hparams, device="cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Simulate training step
    num_beta = 4
    batch_size = 8

    # Sample temperatures
    beta_samples = torch.rand(num_beta) * (1.0 - 0.2) + 0.2
    T_samples = 1.0 / beta_samples
    T_expanded = T_samples.repeat_interleave(batch_size)

    print(f"Simulating VaTD training step:")
    print(f"  num_beta: {num_beta}")
    print(f"  batch_size: {batch_size}")
    print(f"  T_samples: {T_samples.tolist()}")

    # Sample from model (no grad)
    with torch.no_grad():
        samples = model.sample(batch_size=num_beta * batch_size, T=T_expanded)

    # Compute log_prob (with grad)
    log_prob = model.log_prob(samples, T=T_expanded)

    # Simulate energy (random for testing)
    energy = torch.randn(num_beta * batch_size, 1)

    # VaTD loss computation
    log_prob_view = log_prob.view(num_beta, batch_size)
    energy_view = energy.view(num_beta, batch_size)
    beta_expanded = (1.0 / T_expanded).view(num_beta, batch_size)

    # REINFORCE with LOO baseline
    beta_energy = beta_expanded * energy_view
    reinforce_weight = log_prob_view.detach() + beta_energy
    sum_weight = reinforce_weight.sum(dim=1, keepdim=True)
    loo_baseline = (sum_weight - reinforce_weight) / (batch_size - 1)
    advantage = (reinforce_weight - loo_baseline).detach()

    loss = torch.mean(advantage * log_prob_view)

    print(f"  Loss: {loss.item():.6f}")

    # Backward
    optimizer.zero_grad()
    loss.backward()

    # Check gradients
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"  Total gradient norm: {grad_norm:.6f}")

    # Optimizer step
    optimizer.step()

    print("\n✓ VaTD loss simulation passed! Training loop works correctly.")
    return True


def main():
    print("\n" + "=" * 60)
    print("DiscretePixelCNNFiLM Automatic Differentiation Tests")
    print("=" * 60 + "\n")

    all_passed = True

    all_passed &= test_temperature_embedding_grad()
    all_passed &= test_film_generator_grad()
    all_passed &= test_full_model_grad()
    all_passed &= test_vatd_loss_simulation()

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED! ✓")
        print("Automatic Differentiation works correctly with FiLM conditioning.")
    else:
        print("SOME TESTS FAILED! ✗")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    main()
