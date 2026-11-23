# /// script
# requires-python = ">=3.13"
# dependencies = ["onnx"]
# ///
import onnx

input_path = "netvlad_onnx/netvlad.onnx"
output_path = "netvlad_onnx/netvlad_single.onnx"
model = onnx.load(input_path)
onnx.save(model, output_path)
