"""
Detailed debug script to find the source of log_prob discrepancy
"""
import torch
import torch.nn.functional as F
from config import RunConfig

def debug_logdet_components():
    """Debug each component of log_det calculation"""

    # Load config
    config = RunConfig.from_yaml("configs/v0.5/ising_realnvp.yaml")
    config.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Create model
    model = config.create_model().to(config.device)
    model.eval()

    # Test parameters
    batch_size = 4  # Small batch for easier debugging
    T = torch.tensor([1.0] * batch_size).to(config.device)

    print("="*80)
    print("Detailed log_det component analysis")
    print("="*80)

    with torch.no_grad():
        # =================================================================
        # Path 1: sample_and_log_prob
        # =================================================================
        print("\n" + "="*80)
        print("PATH 1: sample_and_log_prob() - z_0 -> x")
        print("="*80)

        # Sample z_0
        bs = batch_size
        h_s, w_s = model.size[0] // 2, model.size[1] // 2
        c_s = model.channel * 4
        z_0 = model.prior.rsample((bs, c_s, h_s, w_s)).to(model.device)

        print(f"\n1. Sample z_0 from N(0,1)")
        print(f"   Shape: {z_0.shape}")
        print(f"   Mean: {z_0.mean().item():.6f}, Std: {z_0.std().item():.6f}")

        # log p(z_0)
        log_prob_z = torch.sum(
            -0.5 * (z_0**2) - 0.5 * torch.tensor(2 * torch.pi).log(),
            dim=[1, 2, 3],
        )
        print(f"\n2. log p(z_0) = -0.5*z^2 - 0.5*log(2π)")
        print(f"   log_prob_z mean: {log_prob_z.mean().item():.4f}")

        # Inverse flow
        T_cond = model._prepare_conditional(T, h_s, w_s)
        log_det_flow = torch.zeros(bs).to(model.device)

        z = z_0.clone()
        print(f"\n3. Inverse flow (z_0 -> x_pre)")
        for layer_idx, layer in enumerate(reversed(model.layers)):
            z = z[:, model.inv_perm_indices, :, :]

            # Get s, t manually to inspect
            x_a = z[:, : layer.st_net.in_channels, :, :]
            x_b = z[:, layer.st_net.in_channels :, :, :]
            s, t = layer.st_net(x_a, T_cond)

            print(f"\n   Layer {layer_idx}:")
            print(f"     s: mean={s.mean().item():.6f}, std={s.std().item():.6f}")
            print(f"     sum(s) per sample: {s.sum(dim=[1,2,3]).mean().item():.4f}")

            z_before = z.clone()
            z, log_det_layer = layer(z, 0, T_cond=T_cond, reverse=True)

            # log_det_layer should just be the increment since we passed 0
            print(f"     log_det from layer: {log_det_layer.mean().item():.4f}")

            log_det_flow = log_det_flow + s.sum(dim=[1, 2, 3])

        print(f"\n   Total log_det_flow: {log_det_flow.mean().item():.4f}")

        # Undo squeeze
        x_pre = model.undo_squeeze(z)
        print(f"\n4. Undo squeeze: {z.shape} -> {x_pre.shape}")
        print(f"   x_pre: mean={x_pre.mean().item():.6f}, std={x_pre.std().item():.6f}")
        print(f"   x_pre: min={x_pre.min().item():.6f}, max={x_pre.max().item():.6f}")

        # Compute log_det_tanh
        log_det_tanh_elements = 2 * (
            torch.abs(x_pre) - torch.log(torch.tensor(2.0)) + F.softplus(-2 * torch.abs(x_pre))
        )
        log_det_tanh = log_det_tanh_elements.sum(dim=[1, 2, 3])
        print(f"\n5. log_det_tanh = 2*log(cosh(x_pre))")
        print(f"   Per element mean: {log_det_tanh_elements.mean().item():.6f}")
        print(f"   Total: {log_det_tanh.mean().item():.4f}")

        # Final log_prob
        log_prob_1 = log_prob_z + log_det_flow - log_det_tanh
        print(f"\n6. log p(x) = log p(z_0) + log_det_flow - log_det_tanh")
        print(f"   = {log_prob_z.mean().item():.4f} + {log_det_flow.mean().item():.4f} - {log_det_tanh.mean().item():.4f}")
        print(f"   = {log_prob_1.mean().item():.4f}")

        # Apply tanh
        x = torch.tanh(x_pre)
        print(f"\n7. x = tanh(x_pre)")
        print(f"   x: min={x.min().item():.6f}, max={x.max().item():.6f}")

        # =================================================================
        # Path 2: forward (using the same x)
        # =================================================================
        print("\n" + "="*80)
        print("PATH 2: forward() - x -> z (using same x from path 1)")
        print("="*80)

        # Clamp
        eps = 1e-6
        x_clamped = torch.clamp(x, -1 + eps, 1 - eps)
        print(f"\n1. Clamp x to [{-1+eps:.6f}, {1-eps:.6f}]")
        print(f"   Clamping changed {(x != x_clamped).sum().item()} elements")

        # atanh
        delta_log_det_elements = -torch.log(1 - x_clamped**2)
        delta_log_det = delta_log_det_elements.sum(dim=[1, 2, 3])
        print(f"\n2. delta_log_det = -sum(log(1 - x^2))")
        print(f"   Per element mean: {delta_log_det_elements.mean().item():.6f}")
        print(f"   Total: {delta_log_det.mean().item():.4f}")

        x_pre_2 = torch.atanh(x_clamped)
        print(f"\n3. x_pre_2 = atanh(x)")
        print(f"   x_pre_2: mean={x_pre_2.mean().item():.6f}, std={x_pre_2.std().item():.6f}")
        print(f"   x_pre vs x_pre_2 diff: {(x_pre - x_pre_2).abs().max().item():.6f}")

        # Squeeze
        x_squeezed = model.squeeze(x_pre_2)
        print(f"\n4. Squeeze: {x_pre_2.shape} -> {x_squeezed.shape}")

        # Forward flow
        log_det_sum = delta_log_det.clone()
        x_flow = x_squeezed.clone()
        T_cond_2 = model._prepare_conditional(T, x_squeezed.shape[2], x_squeezed.shape[3])

        print(f"\n5. Forward flow (x -> z)")
        for layer_idx, layer in enumerate(model.layers):
            # Get s, t manually
            x_a = x_flow[:, : layer.st_net.in_channels, :, :]
            x_b = x_flow[:, layer.st_net.in_channels :, :, :]
            s, t = layer.st_net(x_a, T_cond_2)

            print(f"\n   Layer {layer_idx}:")
            print(f"     s: mean={s.mean().item():.6f}, std={s.std().item():.6f}")
            print(f"     sum(s) per sample: {s.sum(dim=[1,2,3]).mean().item():.4f}")

            x_flow, log_det_sum = layer(x_flow, log_det_sum, T_cond=T_cond_2, reverse=False)
            x_flow = x_flow[:, model.perm_indices, :, :]

        z_2 = x_flow
        print(f"\n   Final z: mean={z_2.mean().item():.6f}, std={z_2.std().item():.6f}")

        # log p(z_2)
        log_prob_z_2 = torch.sum(
            -0.5 * (z_2**2) - 0.5 * torch.tensor(2 * torch.pi).log(),
            dim=[1, 2, 3],
        )
        print(f"\n6. log p(z_2)")
        print(f"   log_prob_z_2 mean: {log_prob_z_2.mean().item():.4f}")

        # Final log_prob
        log_prob_2 = log_prob_z_2 + log_det_sum
        print(f"\n7. log p(x) = log p(z) + log_det_sum")
        print(f"   = {log_prob_z_2.mean().item():.4f} + {log_det_sum.mean().item():.4f}")
        print(f"   = {log_prob_2.mean().item():.4f}")

        # =================================================================
        # Comparison
        # =================================================================
        print("\n" + "="*80)
        print("COMPARISON")
        print("="*80)
        print(f"\nlog_prob from sample_and_log_prob(): {log_prob_1.mean().item():.4f}")
        print(f"log_prob from forward():             {log_prob_2.mean().item():.4f}")
        print(f"Difference:                          {(log_prob_1 - log_prob_2).mean().item():.4f}")

        print(f"\nComponent comparison:")
        print(f"  log_prob_z (path1) vs log_prob_z_2 (path2): {log_prob_z.mean().item():.4f} vs {log_prob_z_2.mean().item():.4f}")
        print(f"  log_det_flow (path1):   {log_det_flow.mean().item():.4f}")
        print(f"  log_det_tanh (path1):   {log_det_tanh.mean().item():.4f}")
        print(f"  delta_log_det (path2):  {delta_log_det.mean().item():.4f}")
        print(f"  log_det_sum (path2):    {log_det_sum.mean().item():.4f}")

        print(f"\nExpected relationships:")
        print(f"  log_det_tanh should ≈ delta_log_det: {log_det_tanh.mean().item():.4f} vs {delta_log_det.mean().item():.4f}")
        print(f"    Difference: {(log_det_tanh - delta_log_det).mean().item():.4f}")

if __name__ == "__main__":
    debug_logdet_components()
