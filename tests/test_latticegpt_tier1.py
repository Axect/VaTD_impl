"""
Tests for LatticeGPT Tier 1 engineering fixes (v0.20.1).

Covers:
  1. Shape correctness for sample() and log_prob()
  2. KV-cache correctness (incremental ≡ full-sequence logits)
  3. Periodic PBC position bias
  4. drop_abs_pos_when_rel flag
  5. AMP (bfloat16) stability
  6. analyze_rank.py / analyze_compression.py hook compatibility
"""

import pytest
import torch
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import model_transformer


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

def _make_hparams(**overrides):
    """Base hparams for a small LatticeGPT (fast tests)."""
    hp = dict(
        size=8, category=2, fix_first=1,
        batch_size=4, num_beta=2, beta_min=0.2, beta_max=1.0,
        d_model=32, n_heads=2, n_layers=2, d_ff=64, cond_dim=16,
        dropout=0.0,
        use_rel_pos_bias=True,
        logit_temp_scale=True, temp_scale_min=0.1, temp_scale_max=2.0,
    )
    hp.update(overrides)
    return hp


@pytest.fixture
def model():
    """Small LatticeGPT with rel_pos_bias on CPU."""
    torch.manual_seed(42)
    return model_transformer.LatticeGPT(_make_hparams(), device="cpu")


@pytest.fixture
def model_no_bias():
    """Small LatticeGPT without rel_pos_bias (FlashAttention path)."""
    torch.manual_seed(42)
    return model_transformer.LatticeGPT(
        _make_hparams(use_rel_pos_bias=False), device="cpu"
    )


# ──────────────────────────────────────────────────────────────
# 1. Shape Tests
# ──────────────────────────────────────────────────────────────

class TestShapes:
    def test_sample_shape(self, model):
        T = torch.ones(4)
        with torch.no_grad():
            s = model.sample(batch_size=4, T=T)
        assert s.shape == (4, 1, 8, 8)

    def test_sample_values(self, model):
        T = torch.ones(4)
        with torch.no_grad():
            s = model.sample(batch_size=4, T=T)
        assert set(s.unique().tolist()) <= {-1.0, 1.0}

    def test_sample_fix_first(self, model):
        T = torch.ones(4)
        with torch.no_grad():
            s = model.sample(batch_size=4, T=T)
        assert (s[:, 0, 0, 0] == 1.0).all()

    def test_log_prob_shape(self, model):
        sample = torch.randint(0, 2, (4, 1, 8, 8)).float() * 2 - 1
        sample[:, 0, 0, 0] = 1.0
        T = torch.ones(4)
        lp = model.log_prob(sample, T=T)
        assert lp.shape == (4, 1)

    def test_log_prob_finite_negative(self, model):
        sample = torch.randint(0, 2, (4, 1, 8, 8)).float() * 2 - 1
        sample[:, 0, 0, 0] = 1.0
        T = torch.ones(4)
        lp = model.log_prob(sample, T=T)
        assert torch.isfinite(lp).all()
        assert (lp <= 0).all()

    def test_shapes_no_bias(self, model_no_bias):
        """FlashAttention path (no rel_pos_bias)."""
        T = torch.ones(4)
        sample = torch.randint(0, 2, (4, 1, 8, 8)).float() * 2 - 1
        sample[:, 0, 0, 0] = 1.0
        with torch.no_grad():
            s = model_no_bias.sample(batch_size=4, T=T)
        lp = model_no_bias.log_prob(sample, T=T)
        assert s.shape == (4, 1, 8, 8)
        assert lp.shape == (4, 1)


# ──────────────────────────────────────────────────────────────
# 2. KV-Cache Correctness
# ──────────────────────────────────────────────────────────────

class TestKVCache:
    """Verify incremental decoding produces same logits as full-sequence."""

    @pytest.mark.parametrize("use_bias", [True, False])
    def test_incremental_matches_full(self, use_bias):
        torch.manual_seed(123)
        hp = _make_hparams(use_rel_pos_bias=use_bias)
        model = model_transformer.LatticeGPT(hp, device="cpu")
        model.eval()

        B = 2
        L = 64  # 8×8
        tokens = torch.randint(0, 2, (B, L))
        tokens[:, 0] = 1
        T = torch.ones(B)
        cond = model._get_cond(T)

        # Full-sequence logits
        logits_full = model._forward_logits(tokens, T)

        # Incremental logits (reproducing sample() path)
        tok_emb_all = model.tok_embed(tokens)
        zero_start = torch.zeros(B, 1, model._d_model)
        tok_emb_shifted = torch.cat([zero_start, tok_emb_all[:, :-1, :]], dim=1)

        logits_incr = torch.zeros_like(logits_full)
        past_kvs = None
        for i in range(L):
            x_i = tok_emb_shifted[:, i:i+1, :] + model.pos_embed[i:i+1]
            incr_bias = model._get_rel_pos_bias(query_len=1, kv_len=i + 1)
            out_i, past_kvs = model.backbone(
                x_i, cond, past_kvs=past_kvs, attn_bias=incr_bias
            )
            logits_i = model.output_head(out_i.squeeze(1))
            temp_scale = model._compute_temp_scale(T)
            if isinstance(temp_scale, torch.Tensor):
                logits_i = logits_i * temp_scale.squeeze(-1)
            logits_incr[:, i, :] = logits_i

        max_diff = (logits_full - logits_incr).abs().max().item()
        assert max_diff < 1e-4, f"KV-cache divergence: max_diff={max_diff:.2e}"

    def test_kv_cache_shape(self, model):
        """KV-cache should store [B, n_heads, L, head_dim]."""
        model.eval()
        B = 2
        T = torch.ones(B)
        cond = model._get_cond(T)
        x = torch.randn(B, 1, model._d_model)
        out, kvs = model.backbone(x, cond)
        for k, v in kvs:
            assert k.shape == (B, model.backbone.blocks[0].n_heads, 1,
                               model.backbone.blocks[0].head_dim)
            assert v.shape == k.shape


# ──────────────────────────────────────────────────────────────
# 3. Periodic PBC Position Bias
# ──────────────────────────────────────────────────────────────

class TestPeriodicBias:
    def test_table_size(self, model):
        """Compact table: (H//2+1) × (W//2+1) entries per head."""
        H, W = model.size
        expected_entries = (H // 2 + 1) * (W // 2 + 1)
        n_heads = model.hparams["n_heads"]
        assert model.rel_pos_bias_table.shape == (n_heads, expected_entries)

    def test_index_range(self, model):
        """All indices within [0, table_size)."""
        idx = model.rel_pos_index
        H, W = model.size
        max_idx = (H // 2 + 1) * (W // 2 + 1) - 1
        assert idx.min() >= 0
        assert idx.max() <= max_idx

    def test_index_symmetry(self, model):
        """Periodic distances are symmetric: idx[i,j] == idx[j,i]."""
        idx = model.rel_pos_index
        assert (idx == idx.T).all()

    def test_pbc_nearest_neighbors(self):
        """Sites across PBC boundary should have distance 1."""
        hp = _make_hparams(size=16)
        model = model_transformer.LatticeGPT(hp, device="cpu")
        idx = model.rel_pos_index
        W = 16
        max_dc = W // 2 + 1  # = 9

        # (0,0) and (15,0): dr=min(15,1)=1, dc=0 → idx=1*9+0=9
        pos_0_0, pos_15_0 = 0, 15 * 16
        assert idx[pos_0_0, pos_15_0].item() == 9

        # (0,0) and (0,15): dr=0, dc=min(15,1)=1 → idx=0*9+1=1
        pos_0_15 = 15
        assert idx[pos_0_0, pos_0_15].item() == 1

    def test_pbc_diagonal_corner(self):
        """(0,0) and (15,15) should have distance (1,1) under PBC."""
        hp = _make_hparams(size=16)
        model = model_transformer.LatticeGPT(hp, device="cpu")
        idx = model.rel_pos_index
        max_dc = 16 // 2 + 1  # = 9

        pos_0_0, pos_15_15 = 0, 15 * 16 + 15
        # dr=min(15,1)=1, dc=min(15,1)=1 → idx=1*9+1=10
        assert idx[pos_0_0, pos_15_15].item() == 10

    def test_pbc_max_distance(self):
        """Maximum periodic distance is (H//2, W//2) = (8, 8) for 16×16."""
        hp = _make_hparams(size=16)
        model = model_transformer.LatticeGPT(hp, device="cpu")
        idx = model.rel_pos_index
        max_dc = 16 // 2 + 1  # = 9

        # (0,0) and (8,8): dr=8, dc=8 → idx=8*9+8=80
        pos_8_8 = 8 * 16 + 8
        assert idx[0, pos_8_8].item() == 80

    def test_brute_force_equivalence(self):
        """Compare precomputed index against brute-force calculation."""
        H, W = 8, 8
        hp = _make_hparams(size=8)
        model = model_transformer.LatticeGPT(hp, device="cpu")
        idx = model.rel_pos_index
        max_dc = W // 2 + 1

        L = H * W
        for i in range(L):
            for j in range(L):
                ri, ci = i // W, i % W
                rj, cj = j // W, j % W
                abs_dr = abs(ri - rj)
                abs_dc = abs(ci - cj)
                per_dr = min(abs_dr, H - abs_dr)
                per_dc = min(abs_dc, W - abs_dc)
                expected = per_dr * max_dc + per_dc
                assert idx[i, j].item() == expected, (
                    f"Mismatch at ({ri},{ci})↔({rj},{cj}): "
                    f"got {idx[i,j].item()}, expected {expected}"
                )


# ──────────────────────────────────────────────────────────────
# 4. Drop Absolute Position Embeddings
# ──────────────────────────────────────────────────────────────

class TestDropAbsPos:
    def test_default_keeps_pos_embed(self, model):
        """Default: pos_embed is a learnable Parameter."""
        assert isinstance(model.pos_embed, torch.nn.Parameter)

    def test_drop_makes_zero_buffer(self):
        hp = _make_hparams(drop_abs_pos_when_rel=True)
        model = model_transformer.LatticeGPT(hp, device="cpu")
        assert not isinstance(model.pos_embed, torch.nn.Parameter)
        assert (model.pos_embed == 0).all()

    def test_drop_not_in_parameters(self):
        hp = _make_hparams(drop_abs_pos_when_rel=True)
        model = model_transformer.LatticeGPT(hp, device="cpu")
        param_names = {n for n, _ in model.named_parameters()}
        assert "pos_embed" not in param_names

    def test_no_bias_keeps_pos_embed(self):
        hp = _make_hparams(use_rel_pos_bias=False, drop_abs_pos_when_rel=True)
        model = model_transformer.LatticeGPT(hp, device="cpu")
        # Without rel_pos_bias, pos_embed should remain learnable
        assert isinstance(model.pos_embed, torch.nn.Parameter)

    def test_forward_works_with_drop(self):
        hp = _make_hparams(drop_abs_pos_when_rel=True)
        model = model_transformer.LatticeGPT(hp, device="cpu")
        model.eval()
        sample = torch.randint(0, 2, (2, 1, 8, 8)).float() * 2 - 1
        sample[:, 0, 0, 0] = 1.0
        T = torch.ones(2)
        lp = model.log_prob(sample, T=T)
        assert torch.isfinite(lp).all()


# ──────────────────────────────────────────────────────────────
# 5. AMP (bfloat16) Stability
# ──────────────────────────────────────────────────────────────

class TestAMP:
    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required for AMP test"
    )
    def test_amp_log_prob_no_nan(self):
        """log_prob under bfloat16 autocast produces finite values."""
        hp = _make_hparams(use_amp=True)
        model = model_transformer.LatticeGPT(hp, device="cuda:0")
        model.eval()
        sample = torch.randint(0, 2, (4, 1, 8, 8), device="cuda:0").float() * 2 - 1
        sample[:, 0, 0, 0] = 1.0
        T = torch.ones(4, device="cuda:0")
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            lp = model.log_prob(sample, T=T)
        assert torch.isfinite(lp).all(), f"NaN/Inf in AMP log_prob: {lp}"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required for AMP test"
    )
    def test_amp_backward_no_nan(self):
        """Backward pass under bfloat16 produces finite gradients."""
        hp = _make_hparams(use_amp=True)
        model = model_transformer.LatticeGPT(hp, device="cuda:0")
        model.train()
        sample = torch.randint(0, 2, (4, 1, 8, 8), device="cuda:0").float() * 2 - 1
        sample[:, 0, 0, 0] = 1.0
        T = torch.ones(4, device="cuda:0")
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            lp = model.log_prob(sample, T=T)
            loss = lp.mean()
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), (
                    f"NaN/Inf gradient in {name}"
                )

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required for AMP test"
    )
    def test_amp_high_beta_stability(self):
        """AMP at beta=1.0 (low T, max REINFORCE variance) stays finite."""
        hp = _make_hparams(use_amp=True)
        model = model_transformer.LatticeGPT(hp, device="cuda:0")
        model.train()
        # Low temperature = high beta = max energy magnitude
        T = torch.full((4,), 1.0, device="cuda:0")  # beta = 1.0
        with torch.no_grad():
            sample = model.sample(batch_size=4, T=T)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            lp = model.log_prob(sample, T=T)
            loss = lp.mean()
        loss.backward()
        assert torch.isfinite(loss), f"NaN/Inf loss at beta=1.0: {loss}"

    def test_amp_disabled_on_cpu(self):
        """use_amp=True on CPU should be silently disabled."""
        from util import Trainer
        hp = _make_hparams(use_amp=True)
        model = model_transformer.LatticeGPT(hp, device="cpu")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        trainer = Trainer(
            model, optimizer, scheduler, device="cpu",
            training_mode="sequential", accumulation_steps=1,
        )
        assert not trainer.use_amp


# ──────────────────────────────────────────────────────────────
# 6. Analysis Hook Compatibility
# ──────────────────────────────────────────────────────────────

class TestAnalysisHooks:
    def test_backbone_blocks_accessible(self, model):
        """analyze_rank.py accesses model.backbone.blocks."""
        assert hasattr(model, "backbone")
        assert hasattr(model.backbone, "blocks")
        assert len(model.backbone.blocks) == 2

    def test_forward_hook_captures_output(self, model):
        """Hooks on backbone.blocks[i] capture [B, L, d_model] tensor."""
        model.eval()
        captured = {}

        def make_hook(idx):
            def hook_fn(module, inp, out):
                act = out[0] if isinstance(out, tuple) else out
                captured[idx] = act.detach()
            return hook_fn

        hooks = []
        for i, block in enumerate(model.backbone.blocks):
            hooks.append(block.register_forward_hook(make_hook(i)))

        sample = torch.randint(0, 2, (2, 1, 8, 8)).float() * 2 - 1
        sample[:, 0, 0, 0] = 1.0
        T = torch.ones(2)
        with torch.no_grad():
            model.log_prob(sample, T=T)

        for h in hooks:
            h.remove()

        assert len(captured) == 2
        for idx, act in captured.items():
            assert act.shape == (2, 64, 32), (
                f"Block {idx} output shape {act.shape}, expected (2, 64, 32)"
            )

    def test_named_modules_linear_layers(self, model):
        """W_q/W_k/W_v/W_o should appear as nn.Linear in named_modules."""
        linear_names = [
            name for name, mod in model.named_modules()
            if isinstance(mod, torch.nn.Linear)
            and "backbone.blocks" in name
        ]
        # Each block should have: W_q, W_k, W_v, W_o, ffn.0, ffn.2, adaln1.proj, adaln2.proj
        qkvo_names = [n for n in linear_names if any(
            n.endswith(suffix) for suffix in (".W_q", ".W_k", ".W_v", ".W_o")
        )]
        # 2 blocks × 4 projections = 8
        assert len(qkvo_names) == 8, f"Expected 8 QKV+O, got {len(qkvo_names)}: {qkvo_names}"

    def test_no_multihead_attention(self, model):
        """nn.MultiheadAttention should no longer exist in the model."""
        for name, mod in model.named_modules():
            assert not isinstance(mod, torch.nn.MultiheadAttention), (
                f"Found nn.MultiheadAttention at {name}"
            )


# ──────────────────────────────────────────────────────────────
# 7. Potts compatibility (category > 2)
# ──────────────────────────────────────────────────────────────

class TestPottsCompat:
    def test_potts3_shapes(self):
        hp = _make_hparams(category=3, fix_first=0)
        model = model_transformer.LatticeGPT(hp, device="cpu")
        model.eval()
        T = torch.ones(2)
        with torch.no_grad():
            s = model.sample(batch_size=2, T=T)
        assert s.shape == (2, 1, 8, 8)
        assert set(s.unique().tolist()) <= {0.0, 1.0, 2.0}

        lp = model.log_prob(s, T=T)
        assert lp.shape == (2, 1)
        assert torch.isfinite(lp).all()
