#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to create a dummy dictionary for Fairseq preprocessing
"""
import argparse
from pathlib import Path

def create_dummy_dict(output_file="dummy_dict.txt"):
    """
    Create a dummy dictionary for Fairseq preprocessing
    
    Args:
        output_file: Path to output dictionary file
    """
    output_path = Path(output_file)
    
    # Create the output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create the dictionary
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("<UNUSED> 0\n")
        f.write("<PAD> 1\n")
        f.write("</s> 2\n")  # EOS token
        f.write("<UNK> 3\n")
    
    print(f"Dummy dictionary created: {output_file}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Create a dummy dictionary for Fairseq preprocessing")
    parser.add_argument("--output", type=str, default="dummy_dict.txt", help="Output dictionary file path")
    
    args = parser.parse_args()
    
    success = create_dummy_dict(output_file=args.output)
    
    if success:
        print("Dummy dictionary creation complete")
    else:
        print("Dummy dictionary creation failed")
        exit(1)

if __name__ == "__main__":
    main() 