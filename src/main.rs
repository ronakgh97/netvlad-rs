use anyhow::Result;
use candle_core::{Device, Tensor};
use netvlad_rs::{load_model_from_file, run};

const QUERY_IMAGE: &str = "images/query.png";
const SAMPLE_IMAGES: &[&str] = &[
    "images/sample0.png",
    "images/sample1.png",
    "images/sample2.png",
];

fn main() -> Result<()> {
    load_model_from_file("netvlad_onnx/netvlad_single.onnx")?;

    // run the inference & fill sample space
    let mut sample = Vec::with_capacity(SAMPLE_IMAGES.len());
    for &path in SAMPLE_IMAGES {
        let tensor = image_to_tensor(path)?;
        let start = std::time::Instant::now();
        let tensor = run(tensor)?;
        println!("inference for '{path}' took {:?}", start.elapsed());
        sample.push((path, tensor));
    }

    let query = image_to_tensor(QUERY_IMAGE)?;
    let query = run(query)?;

    // ranking
    let query = query.flatten_all()?; // [32768]
    for (path, tensor) in sample.iter() {
        let f = tensor.flatten_all()?; // [32768]
        let similarity = (query.dot(&f)? / (query.norm()? * f.norm()?))?.to_scalar::<f32>()?;
        println!("sample '{path}': matching = {:.2}%", similarity * 100.0);
    }

    Ok(())
}

fn image_to_tensor(path: &str) -> Result<Tensor> {
    let image = image::open(path)?;
    let image_resized = image.resize_exact(480, 640, image::imageops::FilterType::Lanczos3);
    let image_rgb8 = image_resized.to_rgb8();

    let mut data = Vec::with_capacity(3 * 480 * 640);

    // HWC -> CHW
    for channel in 0..3 {
        for pixel in image_rgb8.pixels() {
            data.push(pixel[channel] as f32 / 255.0);
        }
    }

    let tensor = Tensor::from_vec(data, (1, 3, 480, 640), &Device::Cpu)?;

    Ok(tensor)
}
