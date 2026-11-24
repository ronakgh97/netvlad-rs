use anyhow::{Result, anyhow, bail};
use candle_core::{Device, Tensor};
use prost::Message;
use rustc_hash::FxHashMap;
use std::sync::OnceLock;

#[derive(Clone, PartialEq, Message)]
struct ModelProto {
    #[prost(message, optional, tag = "7")]
    graph: Option<GraphProto>,
}

#[derive(Clone, PartialEq, Message)]
struct GraphProto {
    #[prost(message, repeated, tag = "5")]
    initializer: Vec<TensorProto>,
}

#[derive(Clone, PartialEq, Message)]
struct TensorProto {
    #[prost(string, tag = "8")]
    name: String,
    #[prost(int64, repeated, tag = "1")]
    dims: Vec<i64>,
    #[prost(int32, tag = "2")]
    data_type: i32,
    #[prost(float, repeated, tag = "4")]
    float32_data: Vec<f32>,
    #[prost(int64, repeated, tag = "7")]
    int64_data: Vec<i64>,
    #[prost(bytes = "vec", tag = "9")]
    raw_data: Vec<u8>,
}

static INITIALIZERS: OnceLock<FxHashMap<String, Tensor>> = OnceLock::new();
fn decode_model(bytes: &[u8]) -> Result<FxHashMap<String, Tensor>> {
    let model = ModelProto::decode(bytes)?;
    let graph = model
        .graph
        .ok_or_else(|| anyhow!("graph is missing from protobuf file"))?;

    let mut initializers = FxHashMap::with_capacity_and_hasher(38, Default::default());
    for tensor_proto in graph.initializer {
        let numel = tensor_proto
            .dims
            .iter()
            .try_fold(1usize, |a, &d| {
                usize::try_from(d).ok().and_then(|d| a.checked_mul(d))
            })
            .ok_or_else(|| {
                anyhow!(
                    "tensor '{}': dims {:?} overflow element count",
                    tensor_proto.name,
                    tensor_proto.dims
                )
            })?;

        let shape: Vec<usize> = tensor_proto
            .dims
            .iter()
            .map(|&d| usize::try_from(d).map_err(|_| anyhow!("invalid dim {}", d)))
            .collect::<Result<_, _>>()?;

        let tensor = match tensor_proto.data_type {
            1 => {
                if !tensor_proto.raw_data.is_empty() {
                    let (chunks, remainder) = tensor_proto.raw_data.as_chunks::<4>();

                    if !remainder.is_empty() || chunks.len() != numel {
                        bail!("invalid f32 raw_data");
                    }

                    let data: Vec<f32> = chunks.iter().map(|c| f32::from_le_bytes(*c)).collect();

                    Tensor::from_vec(data, shape, &Device::Cpu)?
                } else {
                    Tensor::from_vec(tensor_proto.float32_data, shape, &Device::Cpu)?
                }
            }

            7 => {
                if !tensor_proto.raw_data.is_empty() {
                    let (chunks, remainder) = tensor_proto.raw_data.as_chunks::<8>();

                    if !remainder.is_empty() || chunks.len() != numel {
                        bail!("invalid i64 raw_data");
                    }

                    let data: Vec<i64> = chunks.iter().map(|c| i64::from_le_bytes(*c)).collect();

                    Tensor::from_vec(data, shape, &Device::Cpu)?
                } else {
                    Tensor::from_vec(tensor_proto.int64_data, shape, &Device::Cpu)?
                }
            }

            _ => unreachable!("netvlad should only have float32 and int64 initializers"),
        };

        initializers.insert(tensor_proto.name, tensor);
    }
    Ok(initializers)
}

pub fn load_model_from_bytes(bytes: &[u8]) -> Result<()> {
    let initializers = decode_model(bytes)?;
    let _ = INITIALIZERS.set(initializers);
    Ok(())
}

pub fn load_model_from_file(path: &str) -> Result<()> {
    let mmap = unsafe { memmap2::Mmap::map(&std::fs::File::open(path)?)? };
    load_model_from_bytes(&mmap[..])
}

pub fn run(input: Tensor) -> Result<Tensor> {
    let initializers = INITIALIZERS
        .get()
        .ok_or_else(|| anyhow!("model not loaded yet"))?;

    const K: usize = 64;
    const D: usize = 512;

    // convolutional weights and biases
    const ENCODERS: &[(&str, bool)] = &[
        ("encoder.0", false),  // 64
        ("encoder.2", true),   // 64 // max_pool
        ("encoder.5", false),  // 128
        ("encoder.7", true),   // 128 // max_pool
        ("encoder.10", false), // 256
        ("encoder.12", false), // 256
        ("encoder.14", true),  // 256 // max_pool
        ("encoder.17", false), // 512
        ("encoder.19", false), // 512
        ("encoder.21", true),  // 512 // max_pool
        ("encoder.24", false), // 512
        ("encoder.26", false), // 512
        ("encoder.28", false), // 512
    ];

    let get = |name: &str| {
        initializers
            .get(name)
            .ok_or_else(|| anyhow!("initializer '{name}' not found"))
    };

    // x = (i - sub) / div
    let mut x = input.broadcast_sub(get("view")?)?;
    x = x.broadcast_div(get("view_1")?)?;

    // VGG16 encoders: 3x3 kernels, stride 1, padding 1 + ReLU + max pooling 2x2 stride 2
    for (name, is_pool) in ENCODERS {
        let wgt = get(&format!("{name}.weight"))?;
        let bia = get(&format!("{name}.bias"))?;
        let bia = bia.reshape((1, bia.dim(0)?, 1, 1))?;

        let mut conv = x.conv2d(wgt, 1, 1, 1, 1)?;
        conv = conv.broadcast_add(&bia)?; // conv_2d does not add bias
        x = conv.relu()?;
        if *is_pool {
            x = x.max_pool2d((2, 2))?;
        }
    }

    let n = x.dim(2)? * x.dim(3)?; // x shape is [1, 512, 30, 40]

    // per-pixel L2 over channels: reduceL2(axis=1) + clip + div
    let norm = x
        .sqr()?
        .sum_keepdim(1)?
        .sqrt()?
        .clamp(9.999999960041972e-13, f32::MAX)?;
    let xn = x.broadcast_div(&norm)?;

    // conv + reshape ([1, 512, 30, 40] -> [1, 64, 1200])
    let conv = xn.conv2d(get("pool.conv.weight")?, 0, 1, 1, 1)?;
    let r = conv.reshape(&[1, K, n])?;

    // softmax (axis=1)
    let s = {
        let max = r.max_keepdim(1)?;
        let exp = r.broadcast_sub(&max)?.exp()?;
        let sum = exp.sum_keepdim(1)?;
        exp.broadcast_div(&sum)? // [1, 64, 1200]
    };

    // reshape + transpose ([1, 512, 1200] -> [1, 1200, 512])
    let t = xn.reshape((1, D, n))?.transpose(1, 2)?;

    // s * t ([1, 64, 1200] * [1, 1200, 512] -> [1, 64, 512])
    let bmm = s.matmul(&t)?;

    // reduce sum (axis=-1) + mul
    let rsm = s.sum_keepdim(2)?.broadcast_mul(get("unsqueeze")?)?;

    // residual: bmm - rsm
    let resi = bmm.broadcast_sub(&rsm)?;

    // intra-normalize: residual * 1 / sqrt(sum(residual^2, axis=2) + 0.001)
    let norm = {
        let r_sq = resi.sqr()?;
        let r_sq_sum = r_sq.sum_keepdim(2)?;
        let eps = Tensor::full(0.001f32, r_sq_sum.shape(), r_sq_sum.device())?;
        let rsqrt = r_sq_sum.broadcast_add(&eps)?.sqrt()?.recip()?;
        resi.broadcast_mul(&rsqrt)?
    };

    // final flatten + L2 norm (axis=1)
    let mut v = norm.reshape((1, 64 * 512))?;
    let vl2 = v
        .sqr()?
        .sum_keepdim(1)?
        .sqrt()?
        .clamp(9.999999960041972e-13, f32::MAX)?;
    v = v.broadcast_div(&vl2)?;

    Ok(v)
}

#[test]
fn test_decode_model() {
    load_model_from_file("netvlad_onnx/netvlad_single.onnx").unwrap();
    assert_eq!(INITIALIZERS.get().unwrap().len(), 38);
}

#[test]
fn test_run() {
    use candle_core::Shape;
    load_model_from_file("netvlad_onnx/netvlad_single.onnx").unwrap();
    let input = Tensor::full(0.0f32, &[1, 3, 480, 640], &Device::Cpu).unwrap();
    let output = run(input).unwrap();
    assert_eq!(output.shape(), &Shape::from_dims(&[1, 32768]));
}
