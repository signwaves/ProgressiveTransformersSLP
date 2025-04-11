#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to convert an ONNX sign language model to TensorFlow Lite format
"""
import os
import argparse
import tensorflow as tf
import onnx
from onnx_tf.backend import prepare
from pathlib import Path
import numpy as np

def convert_to_tflite(input_file, output_file, feature_dim=1839, seq_len=50, quantize=True):
    """
    Convert an ONNX sign language model to TensorFlow Lite format
    
    Args:
        input_file: Path to the ONNX model file
        output_file: Path to save the TFLite model
        feature_dim: Dimension of the MediaPipe features
        seq_len: Maximum sequence length
        quantize: Whether to apply quantization to the model
    """
    print(f"Loading ONNX model from {input_file}")
    onnx_model = onnx.load(input_file)
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare the ONNX model for TensorFlow
    print("Converting ONNX model to TensorFlow")
    tf_rep = prepare(onnx_model)
    
    # Save the TensorFlow model
    temp_saved_model_dir = str(output_dir / "temp_saved_model")
    print(f"Saving TensorFlow model to {temp_saved_model_dir}")
    tf_rep.export_graph(temp_saved_model_dir)
    
    # Convert to TensorFlow Lite
    print("Converting TensorFlow model to TensorFlow Lite")
    converter = tf.lite.TFLiteConverter.from_saved_model(temp_saved_model_dir)
    
    # Define a representative dataset generator for quantization
    def representative_dataset():
        # Generate sample input data for quantization calibration
        for _ in range(100):
            # Create random source indices
            src = np.random.randint(0, 1000, size=(1, seq_len)).astype(np.int32)
            
            # Create random target input with zeros (MediaPipe features + counter)
            trg_input = np.zeros((1, seq_len, feature_dim + 1), dtype=np.float32)
            
            # Create source lengths
            src_lengths = np.array([seq_len], dtype=np.int32)
            
            yield [
                src,
                trg_input,
                np.zeros((1, 1, 1, seq_len), dtype=np.float32),  # src_mask
                src_lengths,
                np.zeros((1, 1, seq_len, seq_len), dtype=np.float32)  # trg_mask
            ]
    
    # Apply quantization if requested
    if quantize:
        print("Applying quantization")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Use float16 quantization (less aggressive than int8)
        converter.target_spec.supported_types = [tf.float16]
        
        # Uncomment for int8 quantization (more aggressive, requires representative dataset)
        # converter.representative_dataset = representative_dataset
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # converter.inference_input_type = tf.int8
        # converter.inference_output_type = tf.int8
    
    # Convert the model
    tflite_model = converter.convert()
    
    # Save the TensorFlow Lite model
    print(f"Saving TensorFlow Lite model to {output_file}")
    with open(output_file, "wb") as f:
        f.write(tflite_model)
    
    print(f"Model converted to TensorFlow Lite: {output_file}")
    
    # Clean up temporary directory (optional)
    import shutil
    shutil.rmtree(temp_saved_model_dir, ignore_errors=True)
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Convert ONNX sign language model to TensorFlow Lite")
    parser.add_argument("--input", type=str, required=True, help="Path to the ONNX model file")
    parser.add_argument("--output", type=str, default="Models/tflite/model.tflite", help="Path to save the TFLite model")
    parser.add_argument("--feature-dim", type=int, default=1839, help="Dimension of the MediaPipe features")
    parser.add_argument("--seq-len", type=int, default=50, help="Maximum sequence length")
    parser.add_argument("--no-quantize", action="store_true", help="Disable quantization")
    
    args = parser.parse_args()
    
    convert_to_tflite(
        input_file=args.input,
        output_file=args.output,
        feature_dim=args.feature_dim,
        seq_len=args.seq_len,
        quantize=not args.no_quantize
    )

if __name__ == "__main__":
    main()