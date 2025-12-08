"""
Debug script to verify log_prob consistency between forward() and sample_and_log_prob()
"""
import torch
import sys
from config import RunConfig
from main import create_ising_energy_fn

def test_logprob_consistency():
    """Test that forward() and sample_and_log_prob() give consistent results"""

    # Load config
    config = RunConfig.from_yaml("configs/v0.5/ising_realnvp.yaml")
    config.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Create model
    model = config.create_model().to(config.device)
    model.eval()

    # Test parameters
    batch_size = 16
    T = torch.tensor([1.0] * batch_size).to(config.device)

    print("="*80)
    print("Testing log_prob consistency between forward() and sample_and_log_prob()")
    print("="*80)

    with torch.no_grad():
        # Method 1: sample_and_log_prob (currently used in training)
        log_prob_1, samples = model.sample_and_log_prob(batch_size=batch_size, T=T)

        # Method 2: forward (alternative)
        log_prob_2 = model.log_prob(samples, T=T)

        print(f"\nBatch size: {batch_size}")
        print(f"Sample shape: {samples.shape}")
        print(f"Sample range: [{samples.min().item():.4f}, {samples.max().item():.4f}]")
        print(f"\nlog_prob from sample_and_log_prob():")
        print(f"  Mean: {log_prob_1.mean().item():.4f}")
        print(f"  Std:  {log_prob_1.std().item():.4f}")
        print(f"  Min:  {log_prob_1.min().item():.4f}")
        print(f"  Max:  {log_prob_1.max().item():.4f}")

        print(f"\nlog_prob from forward():")
        print(f"  Mean: {log_prob_2.mean().item():.4f}")
        print(f"  Std:  {log_prob_2.std().item():.4f}")
        print(f"  Min:  {log_prob_2.min().item():.4f}")
        print(f"  Max:  {log_prob_2.max().item():.4f}")

        # Check difference
        diff = (log_prob_1 - log_prob_2).abs()
        print(f"\nAbsolute difference:")
        print(f"  Mean: {diff.mean().item():.6f}")
        print(f"  Max:  {diff.max().item():.6f}")

        if diff.max().item() > 1e-3:
            print("\n⚠️  WARNING: Large discrepancy detected!")
            print("   forward() and sample_and_log_prob() are NOT consistent!")
        else:
            print("\n✓ PASS: forward() and sample_and_log_prob() are consistent")

    print("\n" + "="*80)
    print("Testing actual loss computation")
    print("="*80)

    # Create energy function
    L = 16
    energy_fn = create_ising_energy_fn(L=L, d=2, device=config.device)

    with torch.no_grad():
        # Compute loss as in training
        log_prob, samples = model.sample_and_log_prob(batch_size=batch_size, T=T)

        # Discretize samples for energy
        samples_discrete = torch.sign(samples)
        samples_discrete[samples_discrete == 0] = 1

        energy = energy_fn(samples_discrete)
        beta = (1.0 / T).unsqueeze(-1)

        # Loss per sample
        loss_raw = log_prob + beta * energy

        # Normalize by pixels
        num_pixels = samples.shape[-2] * samples.shape[-1]
        loss_per_pixel = loss_raw / num_pixels

        print(f"\nlog_prob stats:")
        print(f"  Mean: {log_prob.mean().item():.4f}")
        print(f"  Std:  {log_prob.std().item():.4f}")

        print(f"\nEnergy stats:")
        print(f"  Mean: {energy.mean().item():.4f}")
        print(f"  Std:  {energy.std().item():.4f}")

        print(f"\nbeta * energy stats:")
        print(f"  Mean: {(beta * energy).mean().item():.4f}")
        print(f"  Std:  {(beta * energy).std().item():.4f}")

        print(f"\nLoss (log_prob + beta * energy) per pixel:")
        print(f"  Mean: {loss_per_pixel.mean().item():.4f}")
        print(f"  Std:  {loss_per_pixel.std().item():.4f}")

        # Expected range for reasonable values
        expected_log_prob_range = (-1000, -10)  # Reasonable range
        expected_loss_range = (-100, 100)  # Reasonable range per pixel

        if not (expected_log_prob_range[0] < log_prob.mean().item() < expected_log_prob_range[1]):
            print(f"\n⚠️  WARNING: log_prob is outside expected range {expected_log_prob_range}")

        if not (expected_loss_range[0] < loss_per_pixel.mean().item() < expected_loss_range[1]):
            print(f"\n⚠️  WARNING: loss_per_pixel is outside expected range {expected_loss_range}")

if __name__ == "__main__":
    test_logprob_consistency()
