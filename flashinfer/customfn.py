from typing import Optional

import torch

from .jit import FLASHINFER_CSRC_DIR, has_prebuilt_ops, load_cuda_ops
from .utils import get_cuda_stream, register_custom_op, register_fake_op

_custom_fn_module = None


def get_custom_fn_module():
    global _custom_fn_module
    if _custom_fn_module is None:
        if has_prebuilt_ops:
            from . import _kernels

            _custom_fn_module = _kernels
        else:
            _custom_fn_module = load_cuda_ops(
                "customfn",
                [
                    FLASHINFER_CSRC_DIR / "flashinfer_custom_ops.cu",
                    FLASHINFER_CSRC_DIR / "customfn.cu",
                ],
            )
    return _custom_fn_module


@register_custom_op("flashinfer::div_clamp_to", mutates_args=("out",))
def _div_clamp_to(out: torch.Tensor, input: torch.Tensor, scale: torch.Tensor) -> None:
    with input.device as device:  # device guard
        get_custom_fn_module().div_clamp_to(out, input, scale, get_cuda_stream(device))


@register_fake_op("flashinfer::div_clamp_to")
def _div_clamp_to_fake(
    out: torch.Tensor, input: torch.Tensor, scale: torch.Tensor
) -> None:
    pass


def _check_shape_and_dtype(
    input: torch.Tensor, output: torch.Tensor, output_dtype: torch.dtype
):
    assert input.ndim == output.ndim, f"{input.ndim} != {output.ndim}"
    assert input.shape == output.shape, f"{input.shape} != {output.shape}"
    assert input.dtype in {
        torch.bfloat16,
        torch.float16,
    }, f"{input.dtype} != {torch.bfloat16} or {torch.float16}"
    assert output.dtype == output_dtype, f"{output.dtype} != {output_dtype}"


def div_clamp_to(
    input: torch.Tensor,
    scale: torch.Tensor,
    output_dtype: torch.dtype,
    out: torch.Tensor = None,
) -> torch.Tensor:
    r"""Fused div, clamp and to operation.
    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape: (..., hidden_size), dtype: torch.bfloat16 or torch.float16

    scale: torch.Tensor
        Scale tensor, shape: (hidden_size), dtype: torch.float32
    output_dytpe: torch.dtype
        Output dtype: torch.float8_e4m3fn or torch.float8_e5m2.

    out: Optional[torch.Tensor]
        The the output tensor, if specified, the kernel will update this tensor inplace.

    Returns
    -------
    output: torch.Tensor
        Output tensor, shape (..., hidden_size).
    """
    if input.shape[-1] * input.dtype.itemsize % 16 != 0:
        raise ValueError("The pointers must be multiple of 16 bytes.")
    assert scale.dtype == torch.float32, f"{scale.dtype} != {torch.float32}"
    if out is not None:
        _check_shape_and_dtype(input, out, output_dtype)
    else:
        out = torch.empty(
            input.shape,
            device=input.device,
            dtype=output_dtype,
        )
    _div_clamp_to(out, input, scale.flatten())
    return out
