#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to train SentencePiece tokenizers for word and sentence level data
"""
import os
import argparse
import sentencepiece as spm
from pathlib import Path

def train_sentencepiece(input_file, model_prefix, vocab_size, character_coverage=1.0, 
                       model_type='bpe', max_sentence_length=4192, input_sentence_size=1000000):
    """
    Train a SentencePiece tokenizer model
    
    Args:
        input_file: Path to input text file (one sentence per line)
        model_prefix: Output model prefix (model_prefix.model and model_prefix.vocab will be created)
        vocab_size: Vocabulary size
        character_coverage: Character coverage (1.0 means all characters are covered)
        model_type: Model type ('bpe' or 'unigram')
        max_sentence_length: Maximum sentence length
        input_sentence_size: Maximum number of sentences to use for training
    """
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: Input file {input_file} does not exist")
        return False
        
    print(f"Training SentencePiece model: {model_prefix}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Input file: {input_file}")
    
    # Check input file is not empty
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if not content:
            print(f"Error: Input file {input_file} is empty")
            return False
    
    # Create the output directory if it doesn't exist
    output_dir = Path(model_prefix).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train the model
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type=model_type,
        max_sentence_length=max_sentence_length,
        input_sentence_size=input_sentence_size,
        pad_id=1,
        unk_id=3,
        bos_id=-1,
        eos_id=2,
        pad_piece="<PAD>",
        unk_piece="<UNK>",
        eos_piece="</s>",
        normalization_rule_name="identity"
    )
    
    if not Path(f"{model_prefix}.model").exists() or not Path(f"{model_prefix}.vocab").exists():
        print(f"Error: Failed to train SentencePiece model")
        return False
        
    # Convert .vocab file to format needed by fairseq
    vocab_path = Path(f"{model_prefix}.vocab")
    vocab_lines = []
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                token, idx = parts
                vocab_lines.append(f"{token} {idx}")
    
    with open(vocab_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(vocab_lines))
    
    print(f"SentencePiece model and vocabulary files created: {model_prefix}.model, {model_prefix}.vocab")
    return True

def main():
    parser = argparse.ArgumentParser(description="Train SentencePiece tokenizers for ISL dataset")
    parser.add_argument("--input", type=str, required=True, help="Input text file path")
    parser.add_argument("--model-prefix", type=str, required=True, help="Output model prefix")
    parser.add_argument("--vocab-size", type=int, required=True, help="Vocabulary size")
    parser.add_argument("--model-type", type=str, default="bpe", choices=["bpe", "unigram"], help="Model type")
    parser.add_argument("--character-coverage", type=float, default=1.0, help="Character coverage")
    parser.add_argument("--max-sentence-length", type=int, default=4192, help="Maximum sentence length")
    parser.add_argument("--input-sentence-size", type=int, default=1000000, help="Maximum number of training sentences")
    parser.add_argument("--use-gloss", action="store_true", help="Train on gloss files instead of text files")
    
    args = parser.parse_args()
    
    # Handle common case of passing a directory instead of a file
    input_path = Path(args.input)
    if input_path.is_dir():
        # If the input is a directory, look for all.text or all.gloss
        if args.use_gloss:
            input_file = input_path / "all.gloss"
            if not input_file.exists():
                # Try to create the all.gloss file from train, dev, test
                for split in ["train", "dev", "test"]:
                    split_file = input_path / f"{split}.gloss"
                    if split_file.exists():
                        with open(input_file, "a") as f_all:
                            with open(split_file, "r") as f_split:
                                f_all.write(f_split.read())
        else:
            input_file = input_path / "all.text"
            if not input_file.exists():
                # Try to create the all.text file from train, dev, test
                for split in ["train", "dev", "test"]:
                    split_file = input_path / f"{split}.text"
                    if split_file.exists():
                        with open(input_file, "a") as f_all:
                            with open(split_file, "r") as f_split:
                                f_all.write(f_split.read())
        
        args.input = str(input_file)
    
    success = train_sentencepiece(
        input_file=args.input,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        model_type=args.model_type,
        max_sentence_length=args.max_sentence_length,
        input_sentence_size=args.input_sentence_size
    )
    
    if success:
        print("SentencePiece training complete")
    else:
        print("SentencePiece training failed")
        exit(1)

if __name__ == "__main__":
    main() 