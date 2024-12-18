from typing import Optional

import torch

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
    assert output.dtype == output_dtype, f"{output.dtype} != {output_dtype}"
    assert (
        input.shape == output.shape
    ), f"{input.shape} != {output.shape}"
    

def div_clamp_to(input: torch.Tensor, scale: torch.Tensor, output_dtype: torch.dtype, out: torch.Tensor = None) -> torch.Tensor:
    r"""Fused div, clamp and to operation.

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (..., hidden_size).
    
    scale: torch.Tensor
        Scale tensor, shape (hidden_size).

    output_dytpe: torch.dtype
        Output Dtype, torch.float8_e4m3fn or torch.float8_e5m2.
    
    out: Optional[torch.Tensor]
        The the output tensor, if specified, the kernel will update this tensor inplace.

    Returns
    -------
    output: torch.Tensor
        Output tensor, shape (..., hidden_size).
    """
    if input.shape[-1] * input.dtype.itemsize % 16 != 0:
        raise ValueError("The pointers must be multiple of 16 bytes.")
    if out is not None:
        _check_shape_and_dtype(input, out, output_dtype)
    else:
        out = torch.empty(
            input.shape, device=input.device, dtype=output_dtype,
        )
    _kernels.div_clamp_to(out, input, scale)
    return out
