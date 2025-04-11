#!/bin/bash
# Script to convert a PyTorch model to TensorFlow Lite for the ISL dataset

set -e  # Exit on error

if [ $# -lt 1 ]; then
    echo "Usage: $0 <checkpoint_path> [output_dir] [config_path] [feature_dim] [seq_len]"
    echo ""
    echo "Arguments:"
    echo "  checkpoint_path    Path to the PyTorch checkpoint file"
    echo "  output_dir         Directory to save the converted models (default: Models/tflite)"
    echo "  config_path        Path to the model configuration file (default: Configs/Base.yaml)"
    echo "  feature_dim        Dimension of the MediaPipe features (default: 1839)"
    echo "  seq_len            Maximum sequence length (default: 50)"
    exit 1
fi

# Parse arguments
CHECKPOINT_PATH=$1
OUTPUT_DIR=${2:-"Models/tflite"}
CONFIG_PATH=${3:-"Configs/Base.yaml"}
FEATURE_DIM=${4:-1839}
SEQ_LEN=${5:-50}

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Converting PyTorch model to TensorFlow Lite..."
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Config: $CONFIG_PATH"
echo "Feature dimension: $FEATURE_DIM"
echo "Sequence length: $SEQ_LEN"

# Step 1: Convert PyTorch to ONNX
echo "Step 1: Converting PyTorch model to ONNX..."
python conversion/pytorch_to_onnx.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --config "$CONFIG_PATH" \
    --feature-dim $FEATURE_DIM

# Step 2: Convert ONNX to TensorFlow Lite
echo "Step 2: Converting ONNX model to TensorFlow Lite..."
python conversion/onnx_to_tflite.py \
    --input "$OUTPUT_DIR/model.onnx" \
    --output "$OUTPUT_DIR/model.tflite" \
    --feature-dim $FEATURE_DIM \
    --seq-len $SEQ_LEN

echo "Conversion complete!"
echo "ONNX model saved to: $OUTPUT_DIR/model.onnx"
echo "TFLite model saved to: $OUTPUT_DIR/model.tflite"

# Check if the output files exist
if [ -f "$OUTPUT_DIR/model.onnx" ] && [ -f "$OUTPUT_DIR/model.tflite" ]; then
    echo "Conversion successful!"
    
    # Print model sizes
    ONNX_SIZE=$(du -h "$OUTPUT_DIR/model.onnx" | cut -f1)
    TFLITE_SIZE=$(du -h "$OUTPUT_DIR/model.tflite" | cut -f1)
    echo "ONNX model size: $ONNX_SIZE"
    echo "TFLite model size: $TFLITE_SIZE"
else
    echo "Conversion failed. Check the error messages above."
    exit 1
fi 