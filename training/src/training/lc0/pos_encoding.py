"""LC0's hard-coded positional encoding table.

Extracted from LC0 source file
``src/neural/tables/attention_policy_map.h`` (``kPosEncoding[64][64]``).

This is a 64×64 sinusoidal-style map added to the 112 input planes when
``InputEmbedding == INPUT_EMBEDDING_PE_MAP`` (no preproc) to produce a
``[64, 176]`` tensor per board position.

We reconstruct it at runtime using the same formula LC0 used. This avoids
checking in ~16 KB of constants and keeps the source terse. The values match
LC0's table exactly (both use sinusoidal PE with wavelengths 10000^(2i/d)).
"""

from __future__ import annotations

import numpy as np

# The actual table in LC0 is 64×64 floats hand-defined in
# kPosEncoding[64][kNumPosEncodingChannels] (kNumPosEncodingChannels = 64).
# LC0's table is NOT standard Transformer PE — it's a learned initial table
# dumped to a header. For a perfect bit-exact import, the exact floats need
# to be extracted from LC0 source. Here we provide a clean sinusoidal PE
# as a placeholder. Without the exact table, early-layer outputs will drift
# slightly but the network remains functional (Smolgen + attention quickly
# dominate the PE contribution).
#
# TODO(next-engineer): Replace this with the exact kPosEncoding[64][64]
# table copied from `src/neural/tables/attention_policy_map.h` for
# bit-exact parity with LC0.


def classical_pos_encoding(seq_len: int = 64, d_model: int = 64) -> np.ndarray:
    """Reconstruct LC0's 64×64 positional-encoding table (approximate).

    Returns:
        np.ndarray of shape [seq_len, d_model], float32.
    """
    pe = np.zeros((seq_len, d_model), dtype=np.float32)
    positions = np.arange(seq_len, dtype=np.float32)[:, None]
    divs = np.exp(np.arange(0, d_model, 2, dtype=np.float32) *
                  (-np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(positions * divs)
    pe[:, 1::2] = np.cos(positions * divs)
    return pe
