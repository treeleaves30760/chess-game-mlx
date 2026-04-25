"""LC0 weight import & forward-pass support.

The recommended Python-to-C++ integration path is to export the LC0 network
to ONNX with ``lc0 leela2onnx`` and run it with onnxruntime. ONNX-runtime's
CoreML execution provider can map most ops to the Apple Silicon GPU/ANE,
but first-run compilation for BT4 (>700 MB model) can take >10 min; the
CPU provider is the pragmatic default.

Modules:
- ``encoding`` : 112-plane input encoder bit-compatible with LC0's
  ``INPUT_CLASSICAL_112_PLANE`` format (handles en-passant history undo,
  mirror-on-black-to-move, rule50 / castling / edge planes).
- ``move_index`` : bidirectional mapping between ``chess.Move`` and LC0's
  compact 1858-entry policy vector.
- ``onnx_runner`` : thin wrapper around an ONNX network; exposes
  ``forward_fen(fen) -> LC0Output``.
- ``generate_golden`` : CLI to dump ``(input_planes, expected_output)``
  test vectors for C++ backend validation.
- ``validate_against_lc0`` : compares the ONNX runner output to the lc0
  binary (BLAS backend) on 10 canonical FENs.
- ``net_pb2`` / ``reader`` / ``model`` / ``pos_encoding`` : earlier MLX-port
  attempt, now superseded by the ONNX path; kept for reference only.

See `docs/lc0_import.md` for the full story.
"""
