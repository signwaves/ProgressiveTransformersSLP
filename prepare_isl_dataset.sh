#!/bin/bash
# Script to run all the steps to prepare the ISL dataset

set -e  # Exit on error

echo "Setting up the environment..."
# Install required packages
pip install -q sentencepiece mediapipe opencv-python pandas openpyxl tqdm kaggle kagglehub pyyaml

echo "Step 1: Download the ISL-CSLRT dataset..."
python download_isl_dataset.py

echo "Step 2: Process the ISL-CSLRT dataset and extract MediaPipe features..."
python process_isl_dataset.py

echo "Step 3: Create dummy dictionary for Fairseq preprocessing..."
python create_dummy_dict.py --output dummy_dict.txt

echo "Step 4: Create MediaPipe config for Fairseq preprocessing..."
python create_mediapipe_config.py --output mediapipe_config.yaml

echo "Step 5: Train SentencePiece tokenizers..."
# For word-level text
python train_spm.py --input Data/isl_dataset/word_level/all.text \
                   --model-prefix Data/isl_dataset/word_level/spm_text_bpe1000 \
                   --vocab-size 1000

# For word-level gloss
python train_spm.py --input Data/isl_dataset/word_level/all.gloss \
                   --model-prefix Data/isl_dataset/word_level/spm_gloss_bpe1000 \
                   --vocab-size 1000 \
                   --use-gloss

# For sentence-level text
python train_spm.py --input Data/isl_dataset/sentence_level/all.text \
                   --model-prefix Data/isl_dataset/sentence_level/spm_text_bpe4000 \
                   --vocab-size 4000

# For sentence-level gloss
python train_spm.py --input Data/isl_dataset/sentence_level/all.gloss \
                   --model-prefix Data/isl_dataset/sentence_level/spm_gloss_bpe4000 \
                   --vocab-size 4000 \
                   --use-gloss

echo "All preparation steps completed!"
echo "The data has been processed in Phoenix14T format with:"
echo "- .skels files for the skeleton data"
echo "- .gloss files for the gloss annotations"
echo "- .text files for the text translations"
echo "- .files files for the sequence names"
echo "Now you can train the model with: python training.py --config Configs/Base.yaml" 