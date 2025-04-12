# coding: utf-8
"""
Custom data loading module to replace torchtext dependency
"""
import os
import io
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from constants import UNK_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN, TARGET_PAD
from vocabulary import Vocabulary

@dataclass
class Example:
    """Data class to hold a single example with its fields"""
    src: List[str]
    trg: torch.Tensor
    file_paths: str

class Field:
    """Handles tokenization and numericalizing of a field"""
    def __init__(self, init_token: Optional[str] = None,
                 eos_token: Optional[str] = EOS_TOKEN,
                 pad_token: Optional[str] = PAD_TOKEN,
                 unk_token: Optional[str] = UNK_TOKEN,
                 tokenize: callable = str.split,
                 lower: bool = False,
                 batch_first: bool = True,
                 include_lengths: bool = True,
                 preprocessing: Optional[callable] = None,
                 postprocessing: Optional[callable] = None,
                 use_vocab: bool = True,
                 dtype: torch.dtype = torch.float32):
        self.init_token = init_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.tokenize = tokenize
        self.lower = lower
        self.batch_first = batch_first
        self.include_lengths = include_lengths
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing
        self.use_vocab = use_vocab
        self.dtype = dtype
        self.vocab = None

    def process(self, text: str) -> List[str]:
        """Process a text string into a list of tokens"""
        if self.lower:
            text = text.lower()
        tokens = self.tokenize(text)
        if self.preprocessing is not None:
            tokens = self.preprocessing(tokens)
        return tokens

    def numericalize(self, tokens: List[str]) -> torch.Tensor:
        """Convert tokens to indices"""
        if self.use_vocab:
            if self.vocab is None:
                raise ValueError("Vocabulary not set for field")
            indices = [self.vocab.stoi[token] for token in tokens]
            return torch.tensor(indices, dtype=torch.long)
        else:
            return torch.tensor(tokens, dtype=self.dtype)

class SignDataset(Dataset):
    """Custom dataset for sign language production"""
    def __init__(self, path: str, exts: Tuple[str, str, str],
                 fields: Tuple[Field, Field, Field],
                 trg_size: int, skip_frames: int = 1,
                 filter_pred: Optional[callable] = None):
        self.examples = []
        self.fields = dict(zip(['src', 'trg', 'file_paths'], fields))

        src_path, trg_path, file_path = [os.path.expanduser(path + x) for x in exts]

        with io.open(src_path, mode='r', encoding='utf-8') as src_file, \
             io.open(trg_path, mode='r', encoding='utf-8') as trg_file, \
             io.open(file_path, mode='r', encoding='utf-8') as files_file:

            for src_line, trg_line, files_line in zip(src_file, trg_file, files_file):
                src_line = src_line.strip()
                trg_line = trg_line.strip()
                files_line = files_line.strip()

                if not src_line or not trg_line:
                    continue

                # Process target data
                trg_values = [float(x) + 1e-8 for x in trg_line.split()]
                trg_frames = [trg_values[i:i + trg_size]
                             for i in range(0, len(trg_values), trg_size * skip_frames)]

                example = Example(
                    src=self.fields['src'].process(src_line),
                    trg=torch.tensor(trg_frames, dtype=torch.float32),
                    file_paths=files_line
                )

                if filter_pred is None or filter_pred(example):
                    self.examples.append(example)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.examples[idx]
        return {
            'src': self.fields['src'].numericalize(example.src),
            'trg': example.trg,
            'file_paths': example.file_paths
        }

def collate_fn(batch: List[Dict[str, Any]], pad_idx: int) -> Dict[str, Any]:
    """Custom collate function for batching"""
    src_tensors = [x['src'] for x in batch]
    trg_tensors = [x['trg'] for x in batch]
    file_paths = [x['file_paths'] for x in batch]

    # Pad source sequences
    src_lengths = torch.tensor([len(x) for x in src_tensors])
    src_padded = pad_sequence(src_tensors, batch_first=True, padding_value=pad_idx)

    # Stack target tensors
    trg_padded = pad_sequence(trg_tensors, batch_first=True,
                             padding_value=TARGET_PAD)

    return {
        'src': src_padded,
        'src_lengths': src_lengths,
        'trg': trg_padded,
        'file_paths': file_paths
    }

def make_data_loader(dataset: SignDataset,
                     batch_size: int,
                     shuffle: bool = False,
                     pad_idx: int = 1) -> DataLoader:
    """Create a DataLoader for the dataset"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda b: collate_fn(b, pad_idx)
    )