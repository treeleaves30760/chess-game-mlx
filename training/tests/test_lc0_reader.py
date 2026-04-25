"""Tests for the LC0 protobuf reader.

Verifies:
1. The LINEAR16 decoder produces values in the stated min/max range.
2. The reader extracts sensible architecture metadata from a real LC0 file.
3. All expected top-level weights are populated for a BT-style checkpoint.

These tests require ``data/lc0_nets/t1_256_distilled.pb.gz`` to be present.
If not, the tests are skipped (they fetch once to CI but not on first clone).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# Resolve repo root (this file lives at training/tests/test_lc0_reader.py)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SMALL_NET = REPO_ROOT / "data" / "lc0_nets" / "t1_256_distilled.pb.gz"


@pytest.fixture(scope="module")
def lc0_weights():
    """Decoded LC0 weights. Skip if network file is not available."""
    pytest.importorskip("google.protobuf")
    if not SMALL_NET.exists():
        pytest.skip(
            f"LC0 network file not available at {SMALL_NET}. "
            "Download with: curl -L https://storage.lczero.org/files/"
            "networks-contrib/t1-256x10-distilled-swa-2432500.pb.gz "
            f"-o {SMALL_NET}"
        )
    from training.lc0.reader import read_lc0_network

    return read_lc0_network(SMALL_NET)


def test_decode_layer_linear16_range():
    """Decoded values must respect the encoded min/max range."""
    from training.lc0 import net_pb2
    from training.lc0.reader import decode_layer

    # Build a synthetic Layer with LINEAR16, 4 values spanning the full range.
    layer = net_pb2.Weights.Layer()
    layer.encoding = net_pb2.Weights.Layer.LINEAR16
    layer.min_val = -1.5
    layer.max_val = 2.5
    # Four uint16 values: 0, 0xffff/3, 2*0xffff/3, 0xffff
    vals = [0, 0x5555, 0xAAAA, 0xFFFF]
    layer.params = b"".join(v.to_bytes(2, "little") for v in vals)

    out = decode_layer(layer)
    assert out.shape == (4,)
    assert out.dtype == np.float32
    # First should decode to min, last to max
    np.testing.assert_allclose(out[0], -1.5, atol=1e-4)
    np.testing.assert_allclose(out[-1], 2.5, atol=1e-4)
    # Monotone
    assert (np.diff(out) > 0).all()


def test_decode_layer_empty():
    """An unset Layer (no params) should return an empty array."""
    from training.lc0 import net_pb2
    from training.lc0.reader import decode_layer

    layer = net_pb2.Weights.Layer()
    out = decode_layer(layer)
    assert out.size == 0
    assert out.dtype == np.float32


def test_reader_extracts_architecture(lc0_weights) -> None:
    """Architecture fields must be correctly inferred."""
    # t1-256x10-distilled specs
    assert lc0_weights.n_layers == 10
    assert lc0_weights.d_model == 256
    assert lc0_weights.n_heads == 8
    assert lc0_weights.ffn_dim == 1024
    assert lc0_weights.smolgen_hidden_channels == 32
    assert lc0_weights.smolgen_hidden_sz == 256
    assert lc0_weights.smolgen_gen_sz == 256
    # Network structure = 4 = NETWORK_SE_WITH_HEADFORMAT for this net
    assert lc0_weights.network_structure == 4
    # Input format: 112-plane classical
    assert lc0_weights.input_format == 1


def test_reader_populates_all_encoder_layers(lc0_weights) -> None:
    """All 10 encoder layers must have complete weight tensors."""
    assert len(lc0_weights.encoder) == 10
    d = lc0_weights.d_model
    ffn = lc0_weights.ffn_dim
    for i, enc in enumerate(lc0_weights.encoder):
        # MHA weights
        assert enc.mha.q_w.size == d * d, f"Layer {i}: q_w shape"
        assert enc.mha.k_w.size == d * d, f"Layer {i}: k_w shape"
        assert enc.mha.v_w.size == d * d, f"Layer {i}: v_w shape"
        assert enc.mha.dense_w.size == d * d, f"Layer {i}: dense_w shape"
        # Biases
        assert enc.mha.q_b.size == d, f"Layer {i}: q_b shape"
        # LNs
        assert enc.ln1_gammas.size == d
        assert enc.ln2_gammas.size == d
        # FFN
        assert enc.ffn.dense1_w.size == d * ffn, f"Layer {i}: ffn.dense1_w"
        assert enc.ffn.dense2_w.size == ffn * d, f"Layer {i}: ffn.dense2_w"
        # Smolgen
        assert enc.mha.smolgen.has_any(), f"Layer {i}: smolgen must exist"


def test_reader_finds_global_smolgen(lc0_weights) -> None:
    """Global smolgen_w must be present and match gen_sz * 64 * 64."""
    expected = lc0_weights.smolgen_gen_sz * 64 * 64
    assert lc0_weights.smolgen_w.size == expected, (
        f"smolgen_w size {lc0_weights.smolgen_w.size} != {expected}"
    )


def test_reader_finds_input_embedding(lc0_weights) -> None:
    """Input embedding must be populated for T1/T2/T3 (PE_MAP path)."""
    # PE_MAP: ip_emb_w is [fist_stage_out_c=176, d_model]
    assert lc0_weights.ip_emb_w.size == 176 * lc0_weights.d_model
    assert lc0_weights.ip_emb_b.size == lc0_weights.d_model
    # PE_MAP → no preproc
    assert lc0_weights.ip_emb_preproc_w.size == 0


def test_reader_finds_heads(lc0_weights) -> None:
    """Policy, value, and moves-left heads must be populated."""
    # T1 uses legacy single heads (not multi-head)
    p = lc0_weights.policy_head_legacy
    v = lc0_weights.value_head_legacy
    assert p.ip_pol_w.size > 0
    assert p.ip2_pol_w.size > 0
    assert v.ip_val_w.size > 0
    assert v.ip1_val_w.size > 0
    # Moves-left at top level
    assert lc0_weights.ip_mov_w.size > 0
    assert lc0_weights.ip1_mov_w.size > 0


def test_rejects_onnx_networks(tmp_path):
    """ONNX-only networks must be rejected with a clear error."""
    import gzip

    from training.lc0 import net_pb2
    from training.lc0.reader import read_lc0_network

    # Build a minimal ONNX-only Net proto
    net = net_pb2.Net()
    net.onnx_model.model = b"dummy"
    net.onnx_model.data_type = net_pb2.OnnxModel.FLOAT
    serialized = net.SerializeToString()

    out = tmp_path / "onnx.pb.gz"
    with gzip.open(out, "wb") as f:
        f.write(serialized)

    with pytest.raises(ValueError, match="ONNX"):
        read_lc0_network(out)
