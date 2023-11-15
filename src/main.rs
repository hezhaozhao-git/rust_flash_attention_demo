use candle_ext::{candle::{D, DType, Device, Result, Tensor}, TensorExt, F};

use candle_flash_attn_v1::flash_attn_varlen;

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let q = Tensor::randn(0., 1., (3, 3, 2, 4), &device)?;
    let k = Tensor::randn(0., 1., (3, 3, 2, 4), &device)?;
    let v = Tensor::randn(0., 1., (3, 3, 2, 4), &device)?;
    let m = Tensor::ones((q.dim(D::Minus2)?, k.dim(D::Minus2)?), DType::U8, &device)?.tril(0)?;

    let o = F::scaled_dot_product_attention(&q, &k, &v, Some(&m), None, None, Some(1.0))?;
    println!("{:?}", o);

    let dims = q.dims().to_vec();
    let (batch_size, seq_len, _n_heads, _d) = (dims[0], dims[1], dims[2], dims[3]);
    let seqlens_q = Tensor::arange_step(0f32, (batch_size as f32 + 1.0) * seq_len as f32, seq_len as f32, &device)?;
    let seqlens_k = Tensor::arange_step(0f32, (batch_size as f32 + 1.0) * seq_len as f32, seq_len as f32, &device)?;
    let o1 = flash_attn_varlen(&q, &k, &v, &seqlens_q, &seqlens_k, seq_len , seq_len, 1.0, true);
    println!("{:?}", o1);
    Ok(())
}