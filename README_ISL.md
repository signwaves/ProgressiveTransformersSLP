# Sign Language Translation with ISL-CSLRT Dataset on Kaggle

This guide explains how to train a sign language translation model using the ISL-CSLRT dataset in a Kaggle notebook, all the way to creating a TensorFlow Lite model for mobile deployment.

## Table of Contents

1. [Setting Up Kaggle Notebook](#setting-up-kaggle-notebook)
2. [Clone the Repository](#clone-the-repository)
3. [Download and Process the ISL-CSLRT Dataset](#download-and-process-the-isl-cslrt-dataset)
4. [Training the Model](#training-the-model)
5. [Evaluating the Model](#evaluating-the-model)
6. [Converting to TensorFlow Lite](#converting-to-tensorflow-lite)
7. [Troubleshooting](#troubleshooting)

## Setting Up Kaggle Notebook

1. Go to [Kaggle](https://www.kaggle.com) and sign in (or create an account)
2. Click on "Create" in the top-right corner and select "New Notebook"
3. In the notebook settings (top-right gear icon):
   - Set Accelerator to "GPU P100" (for faster training)
   - Set Language to "Python"
   - Set "Internet" to "On" (for Git access)
   - Save settings
   
## Clone the Repository

In the first notebook cell, clone the repository:

```python
!git clone https://github.com/aquaticcalf/text-to-sign.git
%cd text-to-sign
```

Install required dependencies:

```python
!pip install -q torch torchtext sentencepiece mediapipe opencv-python pandas openpyxl tqdm pyyaml onnx onnx-tf tensorflow
```

## Download and Process the ISL-CSLRT Dataset

Since we're on Kaggle, we can directly access the ISL-CSLRT dataset without needing authentication:

```python
# Create data directories
!mkdir -p Data/raw Data/features/mediapipe Data/isl_dataset

# Download the ISL-CSLRT dataset directly (Kaggle datasets are accessible without API key)
!kaggle datasets download -d drblack00/isl-csltr-indian-sign-language-dataset --path Data/raw
!unzip -q Data/raw/isl-csltr-indian-sign-language-dataset.zip -d Data/raw
```

Now run the data processing script to extract MediaPipe features and prepare the data:

```python
# Process the data (this may take some time)
!python process_isl_dataset.py
```

This script will:
1. Extract MediaPipe features from sign language video frames
2. Organize data in Phoenix14T format (.skels, .gloss, .text, .files)
3. Split the data into train/dev/test sets
4. Create symlinks in the Data/tmp directory

Prepare tokenizers and necessary configuration files:

```python
# Create necessary files for training
!python create_dummy_dict.py
!python create_mediapipe_config.py

# Train tokenizers for gloss and text
!python train_spm.py --input Data/isl_dataset/word_level/all.text --model-prefix Data/isl_dataset/word_level/spm_text_bpe1000 --vocab-size 1000
!python train_spm.py --input Data/isl_dataset/word_level/all.gloss --model-prefix Data/isl_dataset/word_level/spm_gloss_bpe1000 --vocab-size 1000 --use-gloss
!python train_spm.py --input Data/isl_dataset/sentence_level/all.text --model-prefix Data/isl_dataset/sentence_level/spm_text_bpe4000 --vocab-size 4000
!python train_spm.py --input Data/isl_dataset/sentence_level/all.gloss --model-prefix Data/isl_dataset/sentence_level/spm_gloss_bpe4000 --vocab-size 4000 --use-gloss
```

## Training the Model

The `Base.yaml` configuration file has been updated to work with the ISL-CSLRT dataset and MediaPipe features. Run the training:

```python
# Train the model (this will take several hours)
!python training.py --config Configs/Base.yaml
```

To monitor the training, you can use TensorBoard by adding the following code in a separate cell:

```python
# Set up TensorBoard
%load_ext tensorboard
%tensorboard --logdir Models/ISL
```

## Evaluating the Model

After training, evaluate the model:

```python
# Evaluate on test set
!python prediction.py --config Configs/Base.yaml --checkpoint Models/ISL/best.ckpt --data Data/tmp/test
```

You can visualize some examples using:

```python
# Visualize predictions
!python plot_videos.py --config Configs/Base.yaml --checkpoint Models/ISL/best.ckpt --data Data/tmp/test --output visualizations
```

## Converting to TensorFlow Lite

To convert the PyTorch model to TensorFlow Lite for mobile deployment:

```python
# Install TensorFlow conversion dependencies if needed
!pip install -q onnx onnx-tf tensorflow

# Make the conversion script executable
!chmod +x conversion/convert_model.sh

# Run the conversion pipeline
!./conversion/convert_model.sh Models/ISL/best.ckpt Models/tflite Configs/Base.yaml 1839 50
```

This script performs the following steps:
1. Converts the PyTorch model to ONNX format
2. Converts the ONNX model to TensorFlow Lite format
3. Applies float16 quantization for better performance

The resulting TFLite model will be saved to `Models/tflite/model.tflite`.

## Downloading the TFLite Model

To download the trained model from Kaggle:

```python
from IPython.display import FileLink
FileLink(r'Models/tflite/model.tflite')
```

## Advanced: Manual Conversion

If you need more control over the conversion process, you can run the individual steps manually:

```python
# Step 1: Convert PyTorch to ONNX
!python conversion/pytorch_to_onnx.py \
    --checkpoint Models/ISL/best.ckpt \
    --output-dir Models/tflite \
    --config Configs/Base.yaml \
    --feature-dim 1839

# Step 2: Convert ONNX to TensorFlow Lite
!python conversion/onnx_to_tflite.py \
    --input Models/tflite/model.onnx \
    --output Models/tflite/model.tflite \
    --feature-dim 1839 \
    --seq-len 50
```

## Troubleshooting

### Memory Issues

If you encounter out of memory errors:

1. Reduce batch size in `Configs/Base.yaml`:
   ```python
   # Read the current config
   with open('Configs/Base.yaml', 'r') as f:
       config = f.read()
   
   # Replace batch size
   config = config.replace('batch_size: 4', 'batch_size: 2')
   
   # Write updated config
   with open('Configs/Base.yaml', 'w') as f:
       f.write(config)
   ```

2. Reduce model complexity:
   ```python
   # Reduce model complexity
   config = config.replace('num_layers: 4', 'num_layers: 2')
   config = config.replace('num_heads: 8', 'num_heads: 4')
   ```

3. Use fewer MediaPipe features:
   ```python
   # Use only essential features (hands and pose, skip face)
   !python process_isl_dataset.py --skip-face
   ```

### Kaggle Notebook Timeouts

Kaggle notebooks have time limits. To handle this:

1. Save checkpoints frequently (already enabled in the configuration)
2. Use the following to continue training from a checkpoint:
   ```python
   !python training.py --config Configs/Base.yaml --continue
   ```

3. You can also commit the notebook to save its state:
   ```python
   # Commit the notebook to save progress
   !git config --global user.email "your-email@example.com"
   !git config --global user.name "Your Name"
   !git add -A
   !git commit -m "Save training progress"
   ```

## License

This project is licensed under the terms of the [LICENSE](LICENSE.md) file included.

## Citation

If you use the ISL-CSLRT dataset in your research, please cite:

```
@dataset{drblack00_2022,
  author = {drblack00},
  title = {ISL-CSLTR Indian Sign Language Dataset},
  year = {2022},
  publisher = {Kaggle},
  url = {https://www.kaggle.com/datasets/drblack00/isl-csltr-indian-sign-language-dataset}
}
``` 