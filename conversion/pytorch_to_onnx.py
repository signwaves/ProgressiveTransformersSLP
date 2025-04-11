#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to convert a PyTorch sign language model to ONNX format
"""
import os
import argparse
import torch
import torch.onnx
import yaml
from pathlib import Path

from model import build_model
from vocabulary import Vocabulary

def convert_to_onnx(checkpoint_path, output_dir, config_path=None, feature_dim=1839):
    """
    Convert a PyTorch sign language model to ONNX format
    
    Args:
        checkpoint_path: Path to the PyTorch checkpoint file
        output_dir: Directory to save the ONNX model
        config_path: Path to the model configuration file
        feature_dim: Dimension of the MediaPipe features
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    if config_path is None:
        config_path = "Configs/Base.yaml"
    
    print(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Override feature dimensions if specified
    if feature_dim is not None:
        cfg['model']['trg_size'] = feature_dim
        print(f"Using feature dimension: {feature_dim}")
    
    # Load source vocabulary
    src_vocab_path = cfg['data']['src_vocab']
    print(f"Loading source vocabulary from {src_vocab_path}")
    src_vocab = Vocabulary(file=src_vocab_path)
    
    # Create target vocabulary (just a placeholder for skeleton data)
    trg_vocab = [None] * (cfg['model']['trg_size'] + 1)
    
    # Build the model
    print("Building model")
    model = build_model(cfg=cfg, src_vocab=src_vocab, trg_vocab=trg_vocab)
    
    # Load the trained model weights
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    # Create sample input for the model
    print("Preparing sample input for conversion")
    batch_size = 1
    max_length = 50  # Adjust based on typical input lengths
    
    # Example source indices
    src = torch.randint(0, len(src_vocab), (batch_size, max_length))
    
    # Example target input with MediaPipe features + counter
    trg_input = torch.zeros(batch_size, max_length, cfg['model']['trg_size'] + 1)
    
    # Example source lengths
    src_lengths = torch.tensor([max_length] * batch_size)
    
    # Export to ONNX
    onnx_file_path = output_dir / "model.onnx"
    print(f"Exporting model to ONNX: {onnx_file_path}")
    
    torch.onnx.export(
        model,
        (src, trg_input, None, src_lengths, None),  # Model inputs as a tuple
        onnx_file_path,
        input_names=['src', 'trg_input', 'src_mask', 'src_lengths', 'trg_mask'],
        output_names=['output'],
        dynamic_axes={
            'src': {0: 'batch_size', 1: 'sequence_length'},
            'trg_input': {0: 'batch_size', 1: 'sequence_length'},
            'output': {0: 'batch_size', 1: 'sequence_length'}
        },
        verbose=True
    )
    
    print(f"Model exported to ONNX: {onnx_file_path}")
    return str(onnx_file_path)

def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch sign language model to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the PyTorch checkpoint file")
    parser.add_argument("--output-dir", type=str, default="Models/tflite", help="Directory to save the ONNX model")
    parser.add_argument("--config", type=str, default=None, help="Path to the model configuration file")
    parser.add_argument("--feature-dim", type=int, default=1839, help="Dimension of the MediaPipe features")
    
    args = parser.parse_args()
    
    convert_to_onnx(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        config_path=args.config,
        feature_dim=args.feature_dim
    )

if __name__ == "__main__":
    main()