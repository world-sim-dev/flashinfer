from typing import Optional

import torch
from .utils import get_cuda_stream

# mypy: disable-error-code="attr-defined"
try:
    from . import _kernels
except ImportError as e:
    import logging
    import os

    if os.environ.get("BUILD_DOC", "0") == "1":
        _kernels = None
        logging.warning("Kernels are not loaded in documentation build mode.")
    else:
        raise e


def _check_shape_and_dtype(input: torch.Tensor, output: torch.Tensor, output_dtype: torch.dtype):
    assert input.ndim == output.ndim, f"{input.ndim} != {output.ndim}"
    assert input.shape == output.shape, f"{input.shape} != {output.shape}"
    assert input.dtype in {torch.bfloat16, torch.float16}, \
        f"{input.dtype} != {torch.bfloat16} or {torch.float16}"
    assert output.dtype == output_dtype, f"{output.dtype} != {output_dtype}"

def div_clamp_to(input: torch.Tensor, scale: torch.Tensor, output_dtype: torch.dtype, out: torch.Tensor = None) -> torch.Tensor:
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
            input.shape, device=input.device, dtype=output_dtype,
        )
    _kernels.div_clamp_to(out, input, scale.flatten(), get_cuda_stream(input.device))
    return out