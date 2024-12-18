import torch
import flashinfer


def test_torch(x, scale, quant_dtype, minn, maxx):
    return (x / scale).clamp(min=minn, max=maxx).to(quant_dtype)

def test_flash(x, scale, quant_dtype):
    return flashinfer.customfn.div_clamp_to(x, scale, quant_dtype)

if __name__ == "__main__":
    quant_dtype = torch.float8_e4m3fn
    input_dtype = torch.bfloat16
    dim = 6144
    batch_size = 1
    seq_len = 8000

    x = torch.randn(batch_size, seq_len, dim).to(0).to(input_dtype)
    scale = torch.randn(dim).to(0).to(input_dtype)

    finfo = torch.finfo(quant_dtype)

    for _ in range(30):
        y_ref = test_torch(x, scale, quant_dtype, finfo.min, finfo.max)

    for _ in range(30):
        y = test_flash(x, scale, quant_dtype)

    assert torch.allclose(
        y_ref.to(torch.float), y.to(torch.float), rtol=1e-3, atol=1e-3
    )