import pytest
import torch
import flashinfer

import torch.nn.functional as F

# pytest -s tests/test_div_clamp_to.py


@pytest.mark.parametrize("quant_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("input_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [128, 256, 512, 2048, 4096, 11008, 16384])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("seq_len", [1, 2, 4, 8, 16, 32, 64, 128, 512])
def test_div_clamp_to(quant_dtype, input_dtype, dim, batch_size, seq_len):
    x = torch.randn(batch_size, seq_len, dim).to(0).to(input_dtype)
    scale = torch.randn(dim).to(0).to(torch.float32)

    finfo = torch.finfo(quant_dtype)
    y_ref = (x.to(torch.float) / scale).clamp(min=finfo.min, max=finfo.max).to(quant_dtype)

    y = flashinfer.customfn.div_clamp_to(x, scale, quant_dtype)

    cos_sim = F.cosine_similarity(y_ref.to(torch.float).reshape(-1), y.to(torch.float).reshape(-1), dim=0)
    assert cos_sim > 0.99

    # assert torch.allclose(
        # y_ref.to(torch.float), y.to(torch.float), rtol=1e-3, atol=1e-3
    # )


@pytest.mark.parametrize("quant_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("input_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [128, 256, 512, 2048, 4096, 11008, 16384])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("seq_len", [1, 2, 4, 8, 16, 32, 64, 128, 512])
def test_div_clamp_to_large_value(quant_dtype, input_dtype, dim, batch_size, seq_len):
    x = (torch.randn(batch_size, seq_len, dim) * 9999 + 1).to(0).to(input_dtype)
    scale = torch.randn(dim).to(0).to(torch.float32)

    finfo = torch.finfo(quant_dtype)
    y_ref = (x.to(torch.float) / scale).clamp(min=finfo.min, max=finfo.max).to(quant_dtype)

    y = flashinfer.customfn.div_clamp_to(x, scale, quant_dtype)

    cos_sim = F.cosine_similarity(y_ref.to(torch.float).reshape(-1), y.to(torch.float).reshape(-1), dim=0)
    assert cos_sim > 0.99

    # assert torch.allclose(
    #     y_ref.to(torch.float), y.to(torch.float), rtol=1e-3, atol=1e-3
    # )


if __name__ == "__main__":
    pytest.main([__file__])
