#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to create the MediaPipe config YAML file for Fairseq
"""
import argparse
from pathlib import Path
import yaml

def create_mediapipe_config(output_file="mediapipe_config.yaml", feature_dim=1839):
    """
    Create a MediaPipe config YAML file for Fairseq
    
    Args:
        output_file: Path to output YAML file
        feature_dim: Feature dimension for MediaPipe features
                     Default is 1839 = 33*4 (pose) + 478*3 (face) + 21*3 (left hand) + 21*3 (right hand)
    """
    output_path = Path(output_file)
    
    # Create the output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create the config
    config = {
        'features': {
            'type': 'mediapipe',
            'dim': feature_dim,
            'normalization': False
        }
    }
    
    # Write the config to a YAML file
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"MediaPipe config created: {output_file}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Create a MediaPipe config YAML file for Fairseq")
    parser.add_argument("--output", type=str, default="mediapipe_config.yaml", help="Output YAML file path")
    parser.add_argument("--feature-dim", type=int, default=1839, help="MediaPipe feature dimension")
    
    args = parser.parse_args()
    
    success = create_mediapipe_config(output_file=args.output, feature_dim=args.feature_dim)
    
    if success:
        print("MediaPipe config creation complete")
    else:
        print("MediaPipe config creation failed")
        exit(1)

if __name__ == "__main__":
    main() 