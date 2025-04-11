#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to process the ISL-CSLRT dataset and extract MediaPipe features for sign language translation
"""
import os
import pandas as pd
import cv2
import numpy as np
import mediapipe as mp
import json
import shutil
from pathlib import Path
from tqdm import tqdm
import openpyxl

# Create needed directories
DATA_DIR = Path('Data')
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
FEATURES_DIR = DATA_DIR / 'features/mediapipe'
ISL_DATASET_DIR = DATA_DIR / 'isl_dataset'

for dataset_type in ['word_level', 'sentence_level']:
    dataset_dir = ISL_DATASET_DIR / dataset_type
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for split in ['train', 'dev', 'test']:
        (dataset_dir / split).mkdir(exist_ok=True)

# MediaPipe initialization
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_mediapipe_features(image_path):
    """Extract MediaPipe features from an image"""
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Warning: Could not read image: {image_path}")
        return None
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        refine_face_landmarks=True
    ) as holistic:
        results = holistic.process(image)
        
        # Extract keypoints
        # Pose landmarks: 33 landmarks with x, y, z, visibility
        pose = []
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                # Scale down the coordinates by dividing by 3 as per Phoenix14T format
                pose.extend([landmark.x/3, landmark.y/3, landmark.z/3, landmark.visibility/3])
        else:
            pose = [0.0] * (33 * 4)
        
        # Face landmarks: 478 landmarks with x, y, z
        face = []
        if results.face_landmarks:
            for landmark in results.face_landmarks.landmark:
                # Scale down the coordinates by dividing by 3
                face.extend([landmark.x/3, landmark.y/3, landmark.z/3])
        else:
            face = [0.0] * (478 * 3)
        
        # Left hand landmarks: 21 landmarks with x, y, z
        left_hand = []
        if results.left_hand_landmarks:
            for landmark in results.left_hand_landmarks.landmark:
                # Scale down the coordinates by dividing by 3
                left_hand.extend([landmark.x/3, landmark.y/3, landmark.z/3])
        else:
            left_hand = [0.0] * (21 * 3)
        
        # Right hand landmarks: 21 landmarks with x, y, z
        right_hand = []
        if results.right_hand_landmarks:
            for landmark in results.right_hand_landmarks.landmark:
                # Scale down the coordinates by dividing by 3
                right_hand.extend([landmark.x/3, landmark.y/3, landmark.z/3])
        else:
            right_hand = [0.0] * (21 * 3)
        
        # Combine all features
        keypoints = pose + face + left_hand + right_hand
        return np.array(keypoints, dtype=np.float32)

def process_word_level(word_frames_dir, word_details_path):
    """Process the word level data from the ISL-CSLRT dataset"""
    print("Processing word level data...")
    
    # Read the metadata file
    try:
        # Try to read the Excel file
        df = pd.read_excel(word_details_path, engine='openpyxl')
    except Exception as e:
        print(f"Error reading word details file: {e}")
        # Fall back to CSV if Excel fails
        try:
            df = pd.read_csv(word_details_path)
        except Exception as e:
            print(f"Error reading word details CSV: {e}")
            return
    
    print(f"Found {len(df)} word entries in metadata file")
    
    # Create split index (train: 80%, dev: 10%, test: 10%)
    word_splits = {}
    unique_words = df['Word'].unique()
    np.random.shuffle(unique_words)
    num_words = len(unique_words)
    train_words = unique_words[:int(0.8 * num_words)]
    dev_words = unique_words[int(0.8 * num_words):int(0.9 * num_words)]
    test_words = unique_words[int(0.9 * num_words):]
    
    # Create dictionary for word splits
    for word in train_words:
        word_splits[word] = 'train'
    for word in dev_words:
        word_splits[word] = 'dev'
    for word in test_words:
        word_splits[word] = 'test'
    
    # Process each word
    split_counter = {'train': 0, 'dev': 0, 'test': 0}
    processed_data = {'train': [], 'dev': [], 'test': []}
    
    # Clear existing files
    for split in ['train', 'dev', 'test']:
        for ext in ['.skels', '.gloss', '.text', '.files']:
            output_file = ISL_DATASET_DIR / 'word_level' / f"{split}{ext}"
            if output_file.exists():
                output_file.unlink()
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing words"):
        word_id = row.get('id') or row.get('ID') or idx
        word = row.get('Word') or row.get('word')
        if not word:
            print(f"Skipping row {idx}, no word found")
            continue
            
        # Determine the split
        split = word_splits.get(word, 'train')  # Default to train if word not found
        
        # Find all frames for this word
        try:
            word_dir = None
            # Check if directory exists with the word ID
            potential_dirs = list(word_frames_dir.glob(f"*{word_id}*"))
            if potential_dirs:
                word_dir = potential_dirs[0]
            
            if not word_dir:
                # Try a more flexible search if exact ID match fails
                potential_dirs = list(word_frames_dir.glob(f"*{word}*"))
                if potential_dirs:
                    word_dir = potential_dirs[0]
            
            if not word_dir:
                print(f"Warning: No frames directory found for word: {word} (ID: {word_id})")
                continue
                
            # Get all frames for this word
            frames = sorted(list(word_dir.glob("*.jpg")) + list(word_dir.glob("*.png")))
            
            if not frames:
                print(f"Warning: No frames found for word: {word} (ID: {word_id})")
                continue
                
            # Extract features for each frame
            all_features = []
            for frame_path in frames:
                features = extract_mediapipe_features(frame_path)
                if features is not None:
                    all_features.append(features)
            
            if not all_features:
                print(f"Warning: Could not extract features for word: {word} (ID: {word_id})")
                continue
                
            # Create sequence ID
            sequence_id = f"{split}/isl_word_{split_counter[split]:04d}"
            split_counter[split] += 1
            
            # Save features file
            output_features_path = FEATURES_DIR / f"{sequence_id.replace('/', '_')}.npy"
            np.save(output_features_path, np.array(all_features))
            
            # Add to processed data
            processed_data[split].append({
                'id': sequence_id,
                'word': word,
                'features_path': str(output_features_path),
                'num_frames': len(all_features)
            })
            
            # Format skeleton data according to Phoenix14T format
            # For each frame: all joint values followed by a counter
            skeleton_data = []
            for i, features in enumerate(all_features):
                # Append all joint values
                skeleton_data.extend(features.tolist())
                # Append counter (frame number)
                skeleton_data.append(i)
            
            # Convert to space-separated string
            skeleton_str = ' '.join(map(str, skeleton_data))
            
            # Write to files
            with open(ISL_DATASET_DIR / 'word_level' / f"{split}.skels", 'a') as f_skels:
                f_skels.write(skeleton_str + '\n')
            
            with open(ISL_DATASET_DIR / 'word_level' / f"{split}.gloss", 'a') as f_gloss:
                f_gloss.write(word + '\n')
                
            with open(ISL_DATASET_DIR / 'word_level' / f"{split}.text", 'a') as f_text:
                f_text.write(word + '\n')
                
            with open(ISL_DATASET_DIR / 'word_level' / f"{split}.files", 'a') as f_files:
                f_files.write(sequence_id + '\n')
                
        except Exception as e:
            print(f"Error processing word {word} (ID: {word_id}): {e}")
    
    # Save all processed data
    with open(ISL_DATASET_DIR / 'word_level' / 'metadata.json', 'w') as f:
        json.dump(processed_data, f, indent=2)
        
    # Create combined files for vocabulary training
    with open(ISL_DATASET_DIR / 'word_level' / 'all.text', 'w') as f_all:
        for split in ['train', 'dev', 'test']:
            with open(ISL_DATASET_DIR / 'word_level' / f"{split}.text", 'r') as f_split:
                f_all.write(f_split.read())
                
    print(f"Word level processing complete: {split_counter['train']} train, {split_counter['dev']} dev, {split_counter['test']} test")

def process_sentence_level(sentence_frames_dir, sentence_details_path):
    """Process the sentence level data from the ISL-CSLRT dataset"""
    print("Processing sentence level data...")
    
    # Read the metadata file
    try:
        # Try to read the Excel file
        df = pd.read_excel(sentence_details_path, engine='openpyxl')
    except Exception as e:
        print(f"Error reading sentence details file: {e}")
        # Fall back to CSV if Excel fails
        try:
            df = pd.read_csv(sentence_details_path)
        except Exception as e:
            print(f"Error reading sentence details CSV: {e}")
            return
    
    print(f"Found {len(df)} sentence entries in metadata file")
    
    # Create split index (train: 80%, dev: 10%, test: 10%)
    indices = np.arange(len(df))
    np.random.shuffle(indices)
    train_indices = indices[:int(0.8 * len(indices))]
    dev_indices = indices[int(0.8 * len(indices)):int(0.9 * len(indices))]
    test_indices = indices[int(0.9 * len(indices)):]
    
    split_indices = {}
    for idx in train_indices:
        split_indices[idx] = 'train'
    for idx in dev_indices:
        split_indices[idx] = 'dev'
    for idx in test_indices:
        split_indices[idx] = 'test'
    
    # Process each sentence
    split_counter = {'train': 0, 'dev': 0, 'test': 0}
    processed_data = {'train': [], 'dev': [], 'test': []}
    
    # Clear existing files
    for split in ['train', 'dev', 'test']:
        for ext in ['.skels', '.gloss', '.text', '.files']:
            output_file = ISL_DATASET_DIR / 'sentence_level' / f"{split}{ext}"
            if output_file.exists():
                output_file.unlink()
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing sentences"):
        sentence_id = row.get('id') or row.get('ID') or idx
        sentence = row.get('Sentence') or row.get('sentence')
        if not sentence:
            print(f"Skipping row {idx}, no sentence found")
            continue
            
        # Determine the split
        split = split_indices.get(idx, 'train')  # Default to train if idx not found
        
        # Find all frames for this sentence
        try:
            sentence_dir = None
            # Check if directory exists with the sentence ID
            potential_dirs = list(sentence_frames_dir.glob(f"*{sentence_id}*"))
            if potential_dirs:
                sentence_dir = potential_dirs[0]
            
            if not sentence_dir:
                # Try searching for first few words of the sentence
                first_words = ' '.join(sentence.split()[:3])
                potential_dirs = list(sentence_frames_dir.glob(f"*{first_words}*"))
                if potential_dirs:
                    sentence_dir = potential_dirs[0]
            
            if not sentence_dir:
                print(f"Warning: No frames directory found for sentence ID: {sentence_id}")
                continue
                
            # Get all frames for this sentence
            frames = sorted(list(sentence_dir.glob("*.jpg")) + list(sentence_dir.glob("*.png")))
            
            if not frames:
                print(f"Warning: No frames found for sentence ID: {sentence_id}")
                continue
                
            # Extract features for each frame
            all_features = []
            for frame_path in frames:
                features = extract_mediapipe_features(frame_path)
                if features is not None:
                    all_features.append(features)
            
            if not all_features:
                print(f"Warning: Could not extract features for sentence ID: {sentence_id}")
                continue
                
            # Create sequence ID
            sequence_id = f"{split}/isl_sentence_{split_counter[split]:04d}"
            split_counter[split] += 1
            
            # Save features file
            output_features_path = FEATURES_DIR / f"{sequence_id.replace('/', '_')}.npy"
            np.save(output_features_path, np.array(all_features))
            
            # Add to processed data
            processed_data[split].append({
                'id': sequence_id,
                'sentence': sentence,
                'features_path': str(output_features_path),
                'num_frames': len(all_features)
            })
            
            # Format skeleton data according to Phoenix14T format
            # For each frame: all joint values followed by a counter
            skeleton_data = []
            for i, features in enumerate(all_features):
                # Append all joint values
                skeleton_data.extend(features.tolist())
                # Append counter (frame number)
                skeleton_data.append(i)
            
            # Convert to space-separated string
            skeleton_str = ' '.join(map(str, skeleton_data))
            
            # Generate gloss by just using capitalized words (simplified approximation)
            # In a real scenario, you would need proper gloss annotations
            gloss = ' '.join([word.upper() for word in sentence.split()])
            
            # Write to files
            with open(ISL_DATASET_DIR / 'sentence_level' / f"{split}.skels", 'a') as f_skels:
                f_skels.write(skeleton_str + '\n')
            
            with open(ISL_DATASET_DIR / 'sentence_level' / f"{split}.gloss", 'a') as f_gloss:
                f_gloss.write(gloss + '\n')
                
            with open(ISL_DATASET_DIR / 'sentence_level' / f"{split}.text", 'a') as f_text:
                f_text.write(sentence + '\n')
                
            with open(ISL_DATASET_DIR / 'sentence_level' / f"{split}.files", 'a') as f_files:
                f_files.write(sequence_id + '\n')
                
        except Exception as e:
            print(f"Error processing sentence ID: {sentence_id}: {e}")
    
    # Save all processed data
    with open(ISL_DATASET_DIR / 'sentence_level' / 'metadata.json', 'w') as f:
        json.dump(processed_data, f, indent=2)
        
    # Create combined files for vocabulary training
    with open(ISL_DATASET_DIR / 'sentence_level' / 'all.text', 'w') as f_all:
        for split in ['train', 'dev', 'test']:
            with open(ISL_DATASET_DIR / 'sentence_level' / f"{split}.text", 'r') as f_split:
                f_all.write(f_split.read())
                
    print(f"Sentence level processing complete: {split_counter['train']} train, {split_counter['dev']} dev, {split_counter['test']} test")

def process_dataset():
    """Process the ISL-CSLRT dataset"""
    print("Processing ISL-CSLRT dataset...")
    
    # Check for dataset paths
    corpus_dir = RAW_DIR / 'ISL_CSLRT_Corpus'
    if not corpus_dir.exists():
        potential_paths = list(RAW_DIR.glob('*'))
        for path in potential_paths:
            if 'ISL' in path.name and path.is_dir():
                corpus_dir = path
                break
    
    if not corpus_dir.exists():
        print("Could not find ISL_CSLRT_Corpus directory.")
        print("Have you downloaded the dataset? Run download_isl_dataset.py first.")
        return False
    
    # Check for word level data
    word_frames_dir = corpus_dir / 'Frames_Word_Level'
    metadata_dir = corpus_dir / 'corpus_csv_files'
    
    # Find word details file
    word_details_path = None
    for path in metadata_dir.glob('*'):
        if 'word' in path.name.lower() and ('detail' in path.name.lower() or 'meta' in path.name.lower()):
            word_details_path = path
            break
    
    if word_frames_dir.exists() and word_details_path:
        print(f"Found word level data and metadata: {word_details_path}")
        process_word_level(word_frames_dir, word_details_path)
    else:
        print("Word level data or metadata not found")
    
    # Check for sentence level data
    sentence_frames_dir = corpus_dir / 'Frames_Sentence_Level'
    
    # Find sentence details file
    sentence_details_path = None
    for path in metadata_dir.glob('*'):
        if ('sentence' in path.name.lower() or 'frame' in path.name.lower()) and ('detail' in path.name.lower() or 'meta' in path.name.lower()):
            sentence_details_path = path
            break
    
    if sentence_frames_dir.exists() and sentence_details_path:
        print(f"Found sentence level data and metadata: {sentence_details_path}")
        process_sentence_level(sentence_frames_dir, sentence_details_path)
    else:
        print("Sentence level data or metadata not found")
    
    # Verify everything is in the right format - check if we need to pad to 150 joint values
    # Prepare data for the existing code to use
    for dataset_type in ['word_level', 'sentence_level']:
        for split in ['train', 'dev', 'test']:
            # Create symlinks for the expected file extensions
            src_path = ISL_DATASET_DIR / dataset_type / f"{split}.text"
            if src_path.exists():
                dst_path = DATA_DIR / 'tmp' / f"{split}.text"
                if not dst_path.parent.exists():
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                if dst_path.exists() or dst_path.is_symlink():
                    dst_path.unlink()
                os.symlink(src_path.absolute(), dst_path.absolute())
                
            src_path = ISL_DATASET_DIR / dataset_type / f"{split}.gloss"
            if src_path.exists():
                dst_path = DATA_DIR / 'tmp' / f"{split}.gloss"
                if dst_path.exists() or dst_path.is_symlink():
                    dst_path.unlink()
                os.symlink(src_path.absolute(), dst_path.absolute())
                
            src_path = ISL_DATASET_DIR / dataset_type / f"{split}.skels"
            if src_path.exists():
                dst_path = DATA_DIR / 'tmp' / f"{split}.skels"
                if dst_path.exists() or dst_path.is_symlink():
                    dst_path.unlink()
                os.symlink(src_path.absolute(), dst_path.absolute())
                
            src_path = ISL_DATASET_DIR / dataset_type / f"{split}.files"
            if src_path.exists():
                dst_path = DATA_DIR / 'tmp' / f"{split}.files"
                if dst_path.exists() or dst_path.is_symlink():
                    dst_path.unlink()
                os.symlink(src_path.absolute(), dst_path.absolute())
    
    return True

if __name__ == "__main__":
    success = process_dataset()
    
    if success:
        print("ISL-CSLRT dataset processing complete!")
        print("Data is ready in Phoenix14T format with .skels, .gloss, .text, and .files extensions")
        print("Next steps:")
        print("1. Configure the model: Update Configs/Base.yaml to set data paths and model parameters")
        print("2. Train the model: python training.py --config Configs/Base.yaml")
    else:
        print("Dataset processing failed. Please check the error messages above.") 