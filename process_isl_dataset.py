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
import argparse
from typing import List, Dict, Tuple
import random

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

def extract_mediapipe_features(video_path: str, skip_face: bool = False) -> Dict:
    """
    Extract MediaPipe features from a video file.
    
    Args:
        video_path: Path to the video file
        skip_face: Whether to skip face landmarks extraction
        
    Returns:
        Dictionary containing extracted features
    """
    cap = cv2.VideoCapture(video_path)
    features = {
        'pose': [],
        'left_hand': [],
        'right_hand': [],
        'face': []
    }
    
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = holistic.process(frame_rgb)
            
            # Extract features
            if results.pose_landmarks:
                pose_features = []
                for landmark in results.pose_landmarks.landmark:
                    pose_features.extend([landmark.x, landmark.y, landmark.z])
                features['pose'].append(pose_features)
            
            if results.left_hand_landmarks:
                left_hand_features = []
                for landmark in results.left_hand_landmarks.landmark:
                    left_hand_features.extend([landmark.x, landmark.y, landmark.z])
                features['left_hand'].append(left_hand_features)
            
            if results.right_hand_landmarks:
                right_hand_features = []
                for landmark in results.right_hand_landmarks.landmark:
                    right_hand_features.extend([landmark.x, landmark.y, landmark.z])
                features['right_hand'].append(right_hand_features)
            
            if not skip_face and results.face_landmarks:
                face_features = []
                for landmark in results.face_landmarks.landmark:
                    face_features.extend([landmark.x, landmark.y, landmark.z])
                features['face'].append(face_features)
    
    cap.release()
    return features

def process_video(video_path: str, output_dir: str, skip_face: bool = False) -> str:
    """
    Process a single video and save its features.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save features
        skip_face: Whether to skip face landmarks extraction
        
    Returns:
        Path to the saved features file
    """
    features = extract_mediapipe_features(video_path, skip_face)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save features
    output_path = os.path.join(output_dir, f"{Path(video_path).stem}.npy")
    np.save(output_path, features)
    
    return output_path

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

def create_phoenix_format_files(data_dir: str, output_dir: str, split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15)):
    """
    Create Phoenix14T format files from the processed data.
    
    Args:
        data_dir: Directory containing processed features
        output_dir: Directory to save Phoenix format files
        split_ratio: Train/Dev/Test split ratio
    """
    # Create output directories
    word_level_dir = os.path.join(output_dir, "word_level")
    sentence_level_dir = os.path.join(output_dir, "sentence_level")
    os.makedirs(word_level_dir, exist_ok=True)
    os.makedirs(sentence_level_dir, exist_ok=True)
    
    # Get all feature files
    feature_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    random.shuffle(feature_files)
    
    # Split data
    n_files = len(feature_files)
    train_end = int(n_files * split_ratio[0])
    dev_end = train_end + int(n_files * split_ratio[1])
    
    train_files = feature_files[:train_end]
    dev_files = feature_files[train_end:dev_end]
    test_files = feature_files[dev_end:]
    
    # Create symlinks
    tmp_dir = os.path.join("Data", "tmp")
    os.makedirs(os.path.join(tmp_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(tmp_dir, "dev"), exist_ok=True)
    os.makedirs(os.path.join(tmp_dir, "test"), exist_ok=True)
    
    # Process each split
    for split, files in [("train", train_files), ("dev", dev_files), ("test", test_files)]:
        # Create symlinks
        for file in files:
            src = os.path.join(data_dir, file)
            dst = os.path.join(tmp_dir, split, file)
            if not os.path.exists(dst):
                os.symlink(os.path.abspath(src), dst)
        
        # Create Phoenix format files
        with open(os.path.join(word_level_dir, f"{split}.skels"), "w") as f:
            f.write("\n".join([f"{file[:-4]}" for file in files]))
        
        with open(os.path.join(word_level_dir, f"{split}.gloss"), "w") as f:
            f.write("\n".join([f"{file[:-4]}" for file in files]))
        
        with open(os.path.join(word_level_dir, f"{split}.text"), "w") as f:
            f.write("\n".join([f"{file[:-4]}" for file in files]))
        
        with open(os.path.join(word_level_dir, f"{split}.files"), "w") as f:
            f.write("\n".join([f"{file[:-4]}" for file in files]))
        
        # Create sentence level files (same as word level for now)
        for ext in [".skels", ".gloss", ".text", ".files"]:
            src = os.path.join(word_level_dir, f"{split}{ext}")
            dst = os.path.join(sentence_level_dir, f"{split}{ext}")
            if not os.path.exists(dst):
                os.symlink(os.path.abspath(src), dst)
    
    # Create all.* files
    for level in ["word_level", "sentence_level"]:
        for ext in [".skels", ".gloss", ".text", ".files"]:
            with open(os.path.join(output_dir, level, f"all{ext}"), "w") as f:
                for split in ["train", "dev", "test"]:
                    with open(os.path.join(output_dir, level, f"{split}{ext}"), "r") as split_file:
                        f.write(split_file.read() + "\n")

def main():
    parser = argparse.ArgumentParser(description="Process ISL-CSLRT dataset")
    parser.add_argument("--skip-face", action="store_true", help="Skip face landmarks extraction")
    parser.add_argument("--data-dir", default="Data/raw", help="Directory containing raw videos")
    parser.add_argument("--output-dir", default="Data/features/mediapipe", help="Directory to save processed features")
    parser.add_argument("--phoenix-dir", default="Data/isl_dataset", help="Directory to save Phoenix format files")
    args = parser.parse_args()
    
    # Process videos
    print("Processing videos...")
    video_files = [f for f in os.listdir(args.data_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    for video_file in tqdm(video_files):
        video_path = os.path.join(args.data_dir, video_file)
        process_video(video_path, args.output_dir, args.skip_face)
    
    # Create Phoenix format files
    print("Creating Phoenix format files...")
    create_phoenix_format_files(args.output_dir, args.phoenix_dir)
    
    print("Done!")

if __name__ == "__main__":
    main() 