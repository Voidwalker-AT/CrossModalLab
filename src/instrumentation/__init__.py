"""Passive activation instrumentation for CrossModalLab (no causal intervention).

Import path:
- ``CaptureSpec`` is CPU-safe (no torch).
- ``capture_passive_residual`` requires torch and is imported lazily.
"""

from src.instrumentation.passive_capture import CaptureSpec

__all__ = [
    "CaptureSpec",
    "capture_passive_residual",
]


def __getattr__(name: str):
    if name == "PassiveCaptureResult":
        from src.instrumentation.passive_capture import PassiveCaptureResult

        return PassiveCaptureResult
    if name == "capture_passive_residual":
        from src.instrumentation.passive_capture import capture_passive_residual

        return capture_passive_residual
    if name == "MultiLayerCaptureSpec":
        from src.instrumentation.multi_layer_capture import MultiLayerCaptureSpec

        return MultiLayerCaptureSpec
    if name == "capture_multi_layer_passive":
        from src.instrumentation.multi_layer_capture import capture_multi_layer_passive

        return capture_multi_layer_passive
    raise AttributeError(name)
