# coding: utf-8
"""
Data module
"""
import sys
import os
import os.path
import io
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset

from constants import UNK_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN, TARGET_PAD
from vocabulary import build_vocab, Vocabulary
from data_loader import SignDataset, Field, make_data_loader, Example

# Load the Regression Data
# Data format should be parallel .txt files for src, trg and files
# Each line of the .txt file represents a new sequence, in the same order in each file
# src file should contain a new source input on each line
# trg file should contain skeleton data, with each line a new sequence, each frame following on from the previous
# Joint values were divided by 3 to move to the scale of -1 to 1
# Each joint value should be separated by a space; " "
# Each frame is partioned using the known trg_size length, which includes all joints (In 2D or 3D) and the counter
# Files file should contain the name of each sequence on a new line
def load_data(cfg: dict) -> (Dataset, Dataset, Optional[Dataset],
                                  Vocabulary, Vocabulary):
    """
    Load train, dev and optionally test data as specified in configuration.
    Vocabularies are created from the training set with a limit of `voc_limit`
    tokens and a minimum token frequency of `voc_min_freq`
    (specified in the configuration dictionary).

    The training data is filtered to include sentences up to `max_sent_length`
    on source and target side.

    :param data_cfg: configuration dictionary for data
        ("data" part of configuation file)
    :return:
        - train_data: training dataset
        - dev_data: development dataset
        - test_data: testdata set if given, otherwise None
        - src_vocab: source vocabulary extracted from training data
        - trg_vocab: target vocabulary extracted from training data
    """
    data_cfg = cfg["data"]
    # Source, Target and Files postfixes
    src_lang = data_cfg["src"]
    trg_lang = data_cfg["trg"]
    files_lang = data_cfg.get("files", "files")
    # Train, Dev and Test Path
    train_path = data_cfg["train"]
    dev_path = data_cfg["dev"]
    test_path = data_cfg["test"]

    level = "word"
    lowercase = False
    max_sent_length = data_cfg["max_sent_length"]
    # Target size is plus one due to the counter required for the model
    trg_size = cfg["model"]["trg_size"] + 1
    # Skip frames is used to skip a set proportion of target frames, to simplify the model requirements
    skip_frames = data_cfg.get("skip_frames", 1)

    EOS_TOKEN = '</s>'
    tok_fun = lambda s: list(s) if level == "char" else s.split()

    # Source field is a tokenised version of the source words
    src_field = Field(init_token=None, eos_token=EOS_TOKEN,
                     pad_token=PAD_TOKEN, tokenize=tok_fun,
                     batch_first=True, lower=lowercase,
                     unk_token=UNK_TOKEN,
                     include_lengths=True)

    # Files field is just a raw text field
    files_field = Field(use_vocab=False, tokenize=lambda x: x)

    # Creating a regression target field
    # Pad token is a vector of output size, containing the constant TARGET_PAD
    reg_trg_field = Field(use_vocab=False,
                         dtype=torch.float32,
                         batch_first=True,
                         include_lengths=False,
                         pad_token=torch.ones((trg_size,))*TARGET_PAD)

    # Create the Training Data, using the SignProdDataset
    train_data = SignProdDataset(path=train_path,
                                    exts=("." + src_lang, "." + trg_lang, "." + files_lang),
                                    fields=(src_field, reg_trg_field, files_field),
                                    trg_size=trg_size,
                                    skip_frames=skip_frames,
                                    filter_pred=
                                    lambda x: len(vars(x)['src'])
                                    <= max_sent_length
                                    and len(vars(x)['trg'])
                                    <= max_sent_length)

    src_max_size = data_cfg.get("src_voc_limit", sys.maxsize)
    src_min_freq = data_cfg.get("src_voc_min_freq", 1)
    src_vocab_file = data_cfg.get("src_vocab", None)
    src_vocab = build_vocab(field="src", min_freq=src_min_freq,
                            max_size=src_max_size,
                            dataset=train_data, vocab_file=src_vocab_file)

    # Create a target vocab just as big as the required target vector size -
    # So that len(trg_vocab) is # of joints + 1 (for the counter)
    trg_vocab = [None]*trg_size

    # Create the Validation Data
    dev_data = SignProdDataset(path=dev_path,
                                  exts=("." + src_lang, "." + trg_lang, "." + files_lang),
                                  trg_size=trg_size,
                                  fields=(src_field, reg_trg_field, files_field),
                                  skip_frames=skip_frames)

    # Create the Testing Data
    test_data = SignProdDataset(
        path=test_path,
        exts=("." + src_lang, "." + trg_lang, "." + files_lang),
        trg_size=trg_size,
        fields=(src_field, reg_trg_field, files_field),
        skip_frames=skip_frames)

    src_field.vocab = src_vocab

    return train_data, dev_data, test_data, src_vocab, trg_vocab


# pylint: disable=global-at-module-level
global max_src_in_batch, max_tgt_in_batch

# pylint: disable=unused-argument,global-variable-undefined
def token_batch_size_fn(new, count, sofar):
    """Compute batch size based on number of tokens (+padding)."""
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    src_elements = count * max_src_in_batch
    if hasattr(new, 'trg'):  # for monolingual data sets ("translate" mode)
        max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
        tgt_elements = count * max_tgt_in_batch
    else:
        tgt_elements = 0
    return max(src_elements, tgt_elements)


def make_data_iter(dataset: SignDataset,
                   batch_size: int,
                   batch_type: str = "sentence",
                   train: bool = False,
                   shuffle: bool = False) -> torch.utils.data.DataLoader:
    """
    Returns a DataLoader for the dataset.

    :param dataset: SignDataset containing src and optionally trg
    :param batch_size: size of the batches the iterator prepares
    :param batch_type: measure batch size by sentence count or by token count
    :param train: whether it's training time
    :param shuffle: whether to shuffle the data before each epoch
    :return: PyTorch DataLoader
    """
    return make_data_loader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle and train,
        pad_idx=dataset.fields['src'].vocab.stoi[PAD_TOKEN]
    )

# Main Dataset Class
class SignProdDataset(SignDataset):
    """Defines a dataset for sign language production."""

    def __init__(self, path, exts, fields, trg_size, skip_frames=1, **kwargs):
        """Create a TranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            trg_size: Target size for each frame
            skip_frames: Number of frames to skip
            Remaining keyword arguments: Passed to the constructor of data.Dataset.
        """

        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1]), ('file_paths', fields[2])]

        # Call the parent class constructor with the parameters it expects
        super(SignProdDataset, self).__init__(
            path=path,
            exts=exts,
            fields=fields,
            trg_size=trg_size,
            skip_frames=skip_frames,
            **kwargs
        )
