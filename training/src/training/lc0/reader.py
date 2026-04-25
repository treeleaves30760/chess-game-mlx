"""LC0 protobuf reader.

Decodes a serialized ``pblczero::Net`` protobuf (as produced by LC0 training
and distributed on https://storage.lczero.org/files/networks-contrib/) and
returns all tensor values as ``numpy`` arrays with fully resolved shapes.

The decoded weights are stored in LC0's **row-major [in, out]** matrix layout,
matching the convention in LC0's ONNX converter (LC0 uses `MatMul(flow, W)`
i.e. the weight is right-multiplied — so a weight of logical shape `[M, N]`
is stored as `[M*N]` row-major with `M` = input dim and `N` = output dim).

Reference: https://github.com/LeelaChessZero/lc0/blob/master/src/utils/weights_adapter.cc
"""

from __future__ import annotations

import gzip
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from training.lc0 import net_pb2  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Encoding constants mirroring Layer::Encoding in net.proto
# ---------------------------------------------------------------------------
_ENC_UNKNOWN = 0
_ENC_LINEAR16 = 1
_ENC_FLOAT16 = 2
_ENC_BFLOAT16 = 3
_ENC_FLOAT32 = 4  # Not in the current proto but seen in LC0 source

# Network structure enum values
NETWORK_CLASSICAL = 1
NETWORK_SE = 2
NETWORK_CLASSICAL_WITH_HEADFORMAT = 3
NETWORK_SE_WITH_HEADFORMAT = 4
NETWORK_ONNX = 5
NETWORK_ATTENTIONBODY_WITH_HEADFORMAT = 6
NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT = 7
NETWORK_AB_LEGACY_WITH_MULTIHEADFORMAT = 134

# Input embedding enum values
INPUT_EMBEDDING_NONE = 0
INPUT_EMBEDDING_PE_MAP = 1
INPUT_EMBEDDING_PE_DENSE = 2

# Activation enum values
ACT_DEFAULT = 0
ACT_MISH = 1
ACT_RELU = 2
ACT_NONE = 3
ACT_TANH = 4
ACT_SIGMOID = 5
ACT_SELU = 6
ACT_SWISH = 7
ACT_RELU_2 = 8
ACT_SOFTMAX = 9


# ---------------------------------------------------------------------------
# Layer decoder
# ---------------------------------------------------------------------------

def decode_layer(layer: object) -> np.ndarray:
    """Decode a single ``pblczero::Weights::Layer`` into a float32 ``np.ndarray``.

    The returned array is flat (1D). Callers are responsible for reshaping
    according to the known semantics of each field.

    Returns an empty array if the field was not set (common for optional
    fields in older checkpoints or unused heads).

    Args:
        layer: a ``net_pb2.Weights.Layer`` instance (or equivalent).
    """
    # A protobuf field that was never set looks like a Layer with empty params.
    params: bytes = getattr(layer, "params", b"") or b""
    if not params:
        return np.array([], dtype=np.float32)

    encoding = getattr(layer, "encoding", _ENC_LINEAR16) or _ENC_LINEAR16
    min_val = float(getattr(layer, "min_val", 0.0) or 0.0)
    max_val = float(getattr(layer, "max_val", 0.0) or 0.0)

    if encoding == _ENC_LINEAR16:
        # LC0 formula:
        #   theta = u16 / 0xffff
        #   f = min * (1 - theta) + max * theta
        raw = np.frombuffer(params, dtype="<u2")  # little-endian uint16
        theta = raw.astype(np.float32) / float(0xFFFF)
        return min_val * (1.0 - theta) + max_val * theta

    if encoding == _ENC_FLOAT16:
        raw = np.frombuffer(params, dtype="<f2")  # little-endian float16
        return raw.astype(np.float32)

    if encoding == _ENC_BFLOAT16:
        raw = np.frombuffer(params, dtype="<u2")
        # BF16 → FP32: pad with 16 low zero bits
        u32 = raw.astype(np.uint32) << 16
        return u32.view(np.float32).copy()

    if encoding == _ENC_FLOAT32:
        return np.frombuffer(params, dtype="<f4").astype(np.float32)

    if encoding == _ENC_UNKNOWN:
        # LC0 defaults unknown to LINEAR16
        raw = np.frombuffer(params, dtype="<u2")
        theta = raw.astype(np.float32) / float(0xFFFF)
        return min_val * (1.0 - theta) + max_val * theta

    raise ValueError(f"Unknown layer encoding {encoding!r}")


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LC0Smolgen:
    compress: np.ndarray = field(default_factory=lambda: np.array([]))
    dense1_w: np.ndarray = field(default_factory=lambda: np.array([]))
    dense1_b: np.ndarray = field(default_factory=lambda: np.array([]))
    ln1_gammas: np.ndarray = field(default_factory=lambda: np.array([]))
    ln1_betas: np.ndarray = field(default_factory=lambda: np.array([]))
    dense2_w: np.ndarray = field(default_factory=lambda: np.array([]))
    dense2_b: np.ndarray = field(default_factory=lambda: np.array([]))
    ln2_gammas: np.ndarray = field(default_factory=lambda: np.array([]))
    ln2_betas: np.ndarray = field(default_factory=lambda: np.array([]))

    def has_any(self) -> bool:
        return self.compress.size > 0


@dataclass
class LC0MHA:
    q_w: np.ndarray = field(default_factory=lambda: np.array([]))
    q_b: np.ndarray = field(default_factory=lambda: np.array([]))
    k_w: np.ndarray = field(default_factory=lambda: np.array([]))
    k_b: np.ndarray = field(default_factory=lambda: np.array([]))
    v_w: np.ndarray = field(default_factory=lambda: np.array([]))
    v_b: np.ndarray = field(default_factory=lambda: np.array([]))
    dense_w: np.ndarray = field(default_factory=lambda: np.array([]))
    dense_b: np.ndarray = field(default_factory=lambda: np.array([]))
    smolgen: LC0Smolgen = field(default_factory=LC0Smolgen)


@dataclass
class LC0FFN:
    dense1_w: np.ndarray = field(default_factory=lambda: np.array([]))
    dense1_b: np.ndarray = field(default_factory=lambda: np.array([]))
    dense2_w: np.ndarray = field(default_factory=lambda: np.array([]))
    dense2_b: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class LC0EncoderLayer:
    mha: LC0MHA = field(default_factory=LC0MHA)
    ln1_gammas: np.ndarray = field(default_factory=lambda: np.array([]))
    ln1_betas: np.ndarray = field(default_factory=lambda: np.array([]))
    ffn: LC0FFN = field(default_factory=LC0FFN)
    ln2_gammas: np.ndarray = field(default_factory=lambda: np.array([]))
    ln2_betas: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class LC0PolicyHead:
    # Attention-policy fields
    ip_pol_w: np.ndarray = field(default_factory=lambda: np.array([]))
    ip_pol_b: np.ndarray = field(default_factory=lambda: np.array([]))
    ip2_pol_w: np.ndarray = field(default_factory=lambda: np.array([]))
    ip2_pol_b: np.ndarray = field(default_factory=lambda: np.array([]))
    ip3_pol_w: np.ndarray = field(default_factory=lambda: np.array([]))
    ip3_pol_b: np.ndarray = field(default_factory=lambda: np.array([]))
    ip4_pol_w: np.ndarray = field(default_factory=lambda: np.array([]))
    pol_headcount: int = 0
    pol_encoder: list[LC0EncoderLayer] = field(default_factory=list)


@dataclass
class LC0ValueHead:
    ip_val_w: np.ndarray = field(default_factory=lambda: np.array([]))
    ip_val_b: np.ndarray = field(default_factory=lambda: np.array([]))
    ip1_val_w: np.ndarray = field(default_factory=lambda: np.array([]))
    ip1_val_b: np.ndarray = field(default_factory=lambda: np.array([]))
    ip2_val_w: np.ndarray = field(default_factory=lambda: np.array([]))
    ip2_val_b: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class LC0Weights:
    """Fully-decoded LC0 weights, organized by attention-body semantics.

    Fields not present in the source checkpoint will be empty arrays.
    Check ``len(x)`` or ``x.size`` to probe.
    """
    # Format metadata from NetworkFormat
    input_format: int = 0
    output_format: int = 0
    network_structure: int = 0
    policy_format: int = 0
    value_format: int = 0
    moves_left_format: int = 0
    default_activation: int = 0
    smolgen_activation: int = 0
    ffn_activation: int = 0
    input_embedding: int = 0

    # Input embedding (attention body)
    ip_emb_preproc_w: np.ndarray = field(default_factory=lambda: np.array([]))
    ip_emb_preproc_b: np.ndarray = field(default_factory=lambda: np.array([]))
    ip_emb_w: np.ndarray = field(default_factory=lambda: np.array([]))
    ip_emb_b: np.ndarray = field(default_factory=lambda: np.array([]))
    ip_emb_ln_gammas: np.ndarray = field(default_factory=lambda: np.array([]))
    ip_emb_ln_betas: np.ndarray = field(default_factory=lambda: np.array([]))
    ip_mult_gate: np.ndarray = field(default_factory=lambda: np.array([]))
    ip_add_gate: np.ndarray = field(default_factory=lambda: np.array([]))
    ip_emb_ffn: LC0FFN = field(default_factory=LC0FFN)
    ip_emb_ffn_ln_gammas: np.ndarray = field(default_factory=lambda: np.array([]))
    ip_emb_ffn_ln_betas: np.ndarray = field(default_factory=lambda: np.array([]))

    # Encoder stack
    encoder: list[LC0EncoderLayer] = field(default_factory=list)
    headcount: int = 0

    # Global smolgen weight (shared across all encoder layers)
    smolgen_w: np.ndarray = field(default_factory=lambda: np.array([]))

    # Heads — for MultiHead networks, maps name → head
    policy_heads: dict[str, LC0PolicyHead] = field(default_factory=dict)
    value_heads: dict[str, LC0ValueHead] = field(default_factory=dict)

    # Legacy (LegacyWeights) policy / value fields (single-head checkpoints)
    policy_head_legacy: LC0PolicyHead = field(default_factory=LC0PolicyHead)
    value_head_legacy: LC0ValueHead = field(default_factory=LC0ValueHead)

    # Moves-left head (attention variant)
    ip_mov_w: np.ndarray = field(default_factory=lambda: np.array([]))
    ip_mov_b: np.ndarray = field(default_factory=lambda: np.array([]))
    ip1_mov_w: np.ndarray = field(default_factory=lambda: np.array([]))
    ip1_mov_b: np.ndarray = field(default_factory=lambda: np.array([]))
    ip2_mov_w: np.ndarray = field(default_factory=lambda: np.array([]))
    ip2_mov_b: np.ndarray = field(default_factory=lambda: np.array([]))

    # -------------- Inferred sizes --------------
    @property
    def d_model(self) -> int:
        return int(self.ip_emb_b.size) if self.ip_emb_b.size else 0

    @property
    def n_layers(self) -> int:
        return len(self.encoder)

    @property
    def n_heads(self) -> int:
        return int(self.headcount)

    @property
    def ffn_dim(self) -> int:
        if self.encoder and self.encoder[0].ffn.dense1_b.size:
            return int(self.encoder[0].ffn.dense1_b.size)
        return 0

    @property
    def smolgen_hidden_channels(self) -> int:
        """Per-square compression size ("inner_size")."""
        if not self.encoder or not self.encoder[0].mha.smolgen.has_any():
            return 0
        return int(self.encoder[0].mha.smolgen.compress.size) // self.d_model

    @property
    def smolgen_hidden_sz(self) -> int:
        """Hidden size of the smolgen bottleneck."""
        if not self.encoder or not self.encoder[0].mha.smolgen.has_any():
            return 0
        return int(self.encoder[0].mha.smolgen.dense1_b.size)

    @property
    def smolgen_gen_sz(self) -> int:
        """Per-head smolgen generation size."""
        if not self.encoder or not self.encoder[0].mha.smolgen.has_any():
            return 0
        return int(self.encoder[0].mha.smolgen.dense2_b.size) // self.headcount

    @property
    def embedding_dense_size(self) -> int:
        """For PE_DENSE input embedding, extra channels added per square."""
        if self.ip_emb_preproc_b.size == 0:
            return 0
        return int(self.ip_emb_preproc_b.size) // 64


# ---------------------------------------------------------------------------
# Decoder helpers
# ---------------------------------------------------------------------------

def _decode_ffn(pb: object) -> LC0FFN:
    return LC0FFN(
        dense1_w=decode_layer(pb.dense1_w),
        dense1_b=decode_layer(pb.dense1_b),
        dense2_w=decode_layer(pb.dense2_w),
        dense2_b=decode_layer(pb.dense2_b),
    )


def _decode_smolgen(pb: object) -> LC0Smolgen:
    return LC0Smolgen(
        compress=decode_layer(pb.compress),
        dense1_w=decode_layer(pb.dense1_w),
        dense1_b=decode_layer(pb.dense1_b),
        ln1_gammas=decode_layer(pb.ln1_gammas),
        ln1_betas=decode_layer(pb.ln1_betas),
        dense2_w=decode_layer(pb.dense2_w),
        dense2_b=decode_layer(pb.dense2_b),
        ln2_gammas=decode_layer(pb.ln2_gammas),
        ln2_betas=decode_layer(pb.ln2_betas),
    )


def _decode_mha(pb: object) -> LC0MHA:
    return LC0MHA(
        q_w=decode_layer(pb.q_w),
        q_b=decode_layer(pb.q_b),
        k_w=decode_layer(pb.k_w),
        k_b=decode_layer(pb.k_b),
        v_w=decode_layer(pb.v_w),
        v_b=decode_layer(pb.v_b),
        dense_w=decode_layer(pb.dense_w),
        dense_b=decode_layer(pb.dense_b),
        smolgen=_decode_smolgen(pb.smolgen),
    )


def _decode_encoder_layer(pb: object) -> LC0EncoderLayer:
    return LC0EncoderLayer(
        mha=_decode_mha(pb.mha),
        ln1_gammas=decode_layer(pb.ln1_gammas),
        ln1_betas=decode_layer(pb.ln1_betas),
        ffn=_decode_ffn(pb.ffn),
        ln2_gammas=decode_layer(pb.ln2_gammas),
        ln2_betas=decode_layer(pb.ln2_betas),
    )


def _decode_policy_head(pb: object) -> LC0PolicyHead:
    head = LC0PolicyHead(
        ip_pol_w=decode_layer(pb.ip_pol_w),
        ip_pol_b=decode_layer(pb.ip_pol_b),
        ip2_pol_w=decode_layer(pb.ip2_pol_w),
        ip2_pol_b=decode_layer(pb.ip2_pol_b),
        ip3_pol_w=decode_layer(pb.ip3_pol_w),
        ip3_pol_b=decode_layer(pb.ip3_pol_b),
        ip4_pol_w=decode_layer(pb.ip4_pol_w),
        pol_headcount=int(getattr(pb, "pol_headcount", 0) or 0),
    )
    for enc in getattr(pb, "pol_encoder", []) or []:
        head.pol_encoder.append(_decode_encoder_layer(enc))
    return head


def _decode_value_head(pb: object) -> LC0ValueHead:
    return LC0ValueHead(
        ip_val_w=decode_layer(pb.ip_val_w),
        ip_val_b=decode_layer(pb.ip_val_b),
        ip1_val_w=decode_layer(pb.ip1_val_w),
        ip1_val_b=decode_layer(pb.ip1_val_b),
        ip2_val_w=decode_layer(pb.ip2_val_w),
        ip2_val_b=decode_layer(pb.ip2_val_b),
    )


# ---------------------------------------------------------------------------
# Top-level reader
# ---------------------------------------------------------------------------

def read_lc0_network(path: str | Path) -> LC0Weights:
    """Read an LC0 ``.pb.gz`` file and return a fully-decoded ``LC0Weights``.

    Only attention-body networks (BT2/BT3/BT4 and transformer-style T1/T2/T3)
    are currently supported. Raises ``ValueError`` for pure-CNN or ONNX nets.
    """
    path = Path(path)
    with gzip.open(path, "rb") as f:
        data = f.read()

    net = net_pb2.Net()
    net.ParseFromString(data)

    if net.HasField("onnx_model"):
        raise ValueError(
            f"{path} is an ONNX-only network. This reader handles only raw "
            "protobuf weight networks. Use onnxruntime or convert upstream."
        )

    if not net.HasField("weights"):
        raise ValueError(f"{path} contains neither weights nor onnx_model.")

    w = net.weights
    fmt = net.format.network_format if net.HasField("format") else None

    out = LC0Weights()
    if fmt is not None:
        out.input_format = int(fmt.input)
        out.output_format = int(fmt.output)
        out.network_structure = int(fmt.network)
        out.policy_format = int(fmt.policy)
        out.value_format = int(fmt.value)
        out.moves_left_format = int(fmt.moves_left)
        out.default_activation = int(fmt.default_activation)
        out.smolgen_activation = int(fmt.smolgen_activation)
        out.ffn_activation = int(fmt.ffn_activation)
        out.input_embedding = int(fmt.input_embedding)

    # Reject pure-CNN networks — they have no transformer body
    if out.network_structure in (NETWORK_CLASSICAL, NETWORK_SE,
                                 NETWORK_CLASSICAL_WITH_HEADFORMAT,
                                 NETWORK_SE_WITH_HEADFORMAT):
        if len(w.encoder) == 0:
            raise ValueError(
                f"{path} is a CNN-only network ({out.network_structure=}, no "
                f"encoder layers). This reader supports transformer bodies."
            )

    # Input embedding (attention body)
    out.ip_emb_preproc_w = decode_layer(w.ip_emb_preproc_w)
    out.ip_emb_preproc_b = decode_layer(w.ip_emb_preproc_b)
    out.ip_emb_w = decode_layer(w.ip_emb_w)
    out.ip_emb_b = decode_layer(w.ip_emb_b)
    out.ip_emb_ln_gammas = decode_layer(w.ip_emb_ln_gammas)
    out.ip_emb_ln_betas = decode_layer(w.ip_emb_ln_betas)
    out.ip_mult_gate = decode_layer(w.ip_mult_gate)
    out.ip_add_gate = decode_layer(w.ip_add_gate)
    out.ip_emb_ffn = _decode_ffn(w.ip_emb_ffn)
    out.ip_emb_ffn_ln_gammas = decode_layer(w.ip_emb_ffn_ln_gammas)
    out.ip_emb_ffn_ln_betas = decode_layer(w.ip_emb_ffn_ln_betas)

    # Encoder stack
    for enc in w.encoder:
        out.encoder.append(_decode_encoder_layer(enc))
    out.headcount = int(w.headcount or 0)

    # Global smolgen
    out.smolgen_w = decode_layer(w.smolgen_w)

    # Moves-left (attention body path)
    out.ip_mov_w = decode_layer(w.ip_mov_w)
    out.ip_mov_b = decode_layer(w.ip_mov_b)
    out.ip1_mov_w = decode_layer(w.ip1_mov_w)
    out.ip1_mov_b = decode_layer(w.ip1_mov_b)
    out.ip2_mov_w = decode_layer(w.ip2_mov_w)
    out.ip2_mov_b = decode_layer(w.ip2_mov_b)

    # Policy / value — support both legacy and multi-head layouts
    # Legacy single-head (top-level ip_pol_w etc.)
    out.policy_head_legacy = LC0PolicyHead(
        ip_pol_w=decode_layer(w.ip_pol_w),
        ip_pol_b=decode_layer(w.ip_pol_b),
        ip2_pol_w=decode_layer(w.ip2_pol_w),
        ip2_pol_b=decode_layer(w.ip2_pol_b),
        ip3_pol_w=decode_layer(w.ip3_pol_w),
        ip3_pol_b=decode_layer(w.ip3_pol_b),
        ip4_pol_w=decode_layer(w.ip4_pol_w),
        pol_headcount=int(w.pol_headcount or 0),
    )
    for enc in w.pol_encoder:
        out.policy_head_legacy.pol_encoder.append(_decode_encoder_layer(enc))

    out.value_head_legacy = LC0ValueHead(
        ip_val_w=decode_layer(w.ip_val_w),
        ip_val_b=decode_layer(w.ip_val_b),
        ip1_val_w=decode_layer(w.ip1_val_w),
        ip1_val_b=decode_layer(w.ip1_val_b),
        ip2_val_w=decode_layer(w.ip2_val_w),
        ip2_val_b=decode_layer(w.ip2_val_b),
    )

    # Multi-head container
    if w.HasField("policy_heads"):
        ph = w.policy_heads
        # Shared ip_pol_w at the container level (some heads reference these)
        shared_w = decode_layer(ph.ip_pol_w)
        shared_b = decode_layer(ph.ip_pol_b)
        for name in ("vanilla", "optimistic_st", "soft", "opponent"):
            if ph.HasField(name):
                head_pb = getattr(ph, name)
                head = _decode_policy_head(head_pb)
                # If per-head ip_pol_w is empty, use the container-level shared ones
                if head.ip_pol_w.size == 0 and shared_w.size > 0:
                    head.ip_pol_w = shared_w
                    head.ip_pol_b = shared_b
                out.policy_heads[name] = head
    if w.HasField("value_heads"):
        vh = w.value_heads
        for name in ("winner", "q", "st"):
            if vh.HasField(name):
                out.value_heads[name] = _decode_value_head(getattr(vh, name))

    return out


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

def summarize(weights: LC0Weights) -> str:
    """Return a human-readable summary of the LC0 network topology."""
    lines = []
    lines.append("LC0 Network Summary")
    lines.append("=" * 50)
    lines.append(f"  InputFormat:      {weights.input_format}  (1 = CLASSICAL_112_PLANE)")
    lines.append(f"  OutputFormat:     {weights.output_format} (2 = OUTPUT_WDL)")
    lines.append(f"  NetworkStructure: {weights.network_structure}")
    lines.append(f"  PolicyFormat:     {weights.policy_format} (3 = POLICY_ATTENTION)")
    lines.append(f"  ValueFormat:      {weights.value_format}")
    lines.append(f"  MovesLeftFormat:  {weights.moves_left_format}")
    lines.append(f"  DefaultAct:       {weights.default_activation} (1 = MISH)")
    lines.append(f"  SmolgenAct:       {weights.smolgen_activation}")
    lines.append(f"  FFNActivation:    {weights.ffn_activation}")
    lines.append(f"  InputEmbedding:   {weights.input_embedding}")
    lines.append("")
    lines.append("Transformer body")
    lines.append(f"  n_layers:         {weights.n_layers}")
    lines.append(f"  d_model:          {weights.d_model}")
    lines.append(f"  n_heads:          {weights.n_heads}")
    lines.append(f"  ffn_dim:          {weights.ffn_dim}")
    lines.append("")
    lines.append("Smolgen")
    lines.append(f"  hidden_channels:  {weights.smolgen_hidden_channels} (per-square compression)")
    lines.append(f"  hidden_sz:        {weights.smolgen_hidden_sz} (bottleneck)")
    lines.append(f"  gen_sz (per-head):{weights.smolgen_gen_sz}")
    lines.append(f"  global smolgen_w size: {weights.smolgen_w.size}")
    lines.append("")
    lines.append("Input embedding")
    lines.append(f"  ip_emb_w size:         {weights.ip_emb_w.size}")
    lines.append(f"  ip_emb_preproc_w size: {weights.ip_emb_preproc_w.size}")
    lines.append(f"  embedding_dense_size:  {weights.embedding_dense_size}")
    lines.append(f"  ip_mult_gate size:     {weights.ip_mult_gate.size}")
    lines.append(f"  ip_add_gate size:      {weights.ip_add_gate.size}")
    lines.append(f"  ip_emb_ffn.dense1_w size: {weights.ip_emb_ffn.dense1_w.size}")
    lines.append("")
    lines.append("Heads")
    if weights.policy_heads:
        lines.append(f"  policy_heads keys: {sorted(weights.policy_heads.keys())}")
        for k, v in weights.policy_heads.items():
            lines.append(f"    {k}: pol_encoder_layers={len(v.pol_encoder)} "
                         f"ip_pol_w={v.ip_pol_w.size} ip2={v.ip2_pol_w.size}")
    else:
        lpl = weights.policy_head_legacy
        lines.append(f"  policy_legacy: ip_pol_w={lpl.ip_pol_w.size} "
                     f"ip2={lpl.ip2_pol_w.size} pol_encoder_layers={len(lpl.pol_encoder)}")
    if weights.value_heads:
        lines.append(f"  value_heads keys: {sorted(weights.value_heads.keys())}")
    else:
        lvh = weights.value_head_legacy
        lines.append(f"  value_legacy: ip_val_w={lvh.ip_val_w.size} "
                     f"ip1={lvh.ip1_val_w.size} ip2={lvh.ip2_val_w.size}")
    lines.append(f"  moves_left: ip_mov_w={weights.ip_mov_w.size} "
                 f"ip1={weights.ip1_mov_w.size} ip2={weights.ip2_mov_w.size}")
    return "\n".join(lines)
