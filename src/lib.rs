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
    INITIALIZERS
        .set(initializers)
        .map_err(|_| anyhow!("model already loaded"))?;
    Ok(())
}

pub fn load_model_from_file(path: &str) -> Result<()> {
    let bytes =
        std::fs::read(path).map_err(|e| anyhow!("failed to read model file '{}': {}", path, e))?;
    load_model_from_bytes(&bytes)
}

#[test]
fn test_decode_model() {
    load_model_from_file("netvlad_onnx/netvlad_single.onnx").unwrap();
    println!("initializer count: {}", INITIALIZERS.get().unwrap().len());
}
