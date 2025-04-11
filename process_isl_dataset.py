#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to process the ISL-CSLRT dataset videos and extract MediaPipe features for sign language translation
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
import argparse
from typing import List, Dict, Tuple
import random

# Create needed directories
DATA_DIR = Path('Data')
RAW_DIR = DATA_DIR / 'raw'
FEATURES_DIR = DATA_DIR / 'features/mediapipe'
ISL_DATASET_DIR = DATA_DIR / 'isl_dataset'

for dir_path in [DATA_DIR, RAW_DIR, FEATURES_DIR, ISL_DATASET_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

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
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return None
        
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

def process_video(video_path: str, output_dir: str, skip_face: bool = False, output_name: str = None) -> str:
    """
    Process a single video and save its features.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save features
        skip_face: Whether to skip face landmarks extraction
        output_name: Optional name for the output file (without extension)
        
    Returns:
        Path to the saved features file
    """
    print(f"Processing video: {video_path}")
    features = extract_mediapipe_features(video_path, skip_face)
    
    if features is None:
        print(f"Skipping video {video_path} due to error")
        return None
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save features using the provided name or default to video filename
    if output_name:
        output_path = os.path.join(output_dir, f"{output_name}.npy")
    else:
        output_path = os.path.join(output_dir, f"{Path(video_path).stem}.npy")
    
    np.save(output_path, features)
    print(f"Saved features to: {output_path}")
    
    return output_path

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
    if not feature_files:
        print(f"Warning: No feature files found in {data_dir}")
        return
        
    print(f"Found {len(feature_files)} feature files")
    random.shuffle(feature_files)
    
    # Split data
    n_files = len(feature_files)
    train_end = int(n_files * split_ratio[0])
    dev_end = train_end + int(n_files * split_ratio[1])
    
    train_files = feature_files[:train_end]
    dev_files = feature_files[train_end:dev_end]
    test_files = feature_files[dev_end:]
    
    print(f"Splitting data: {len(train_files)} train, {len(dev_files)} dev, {len(test_files)} test")
    
    # Create symlinks
    tmp_dir = os.path.join("Data", "tmp")
    os.makedirs(os.path.join(tmp_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(tmp_dir, "dev"), exist_ok=True)
    os.makedirs(os.path.join(tmp_dir, "test"), exist_ok=True)
    
    # Process each split
    for split, files in [("train", train_files), ("dev", dev_files), ("test", test_files)]:
        print(f"Processing {split} split...")
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
    parser = argparse.ArgumentParser(description="Process ISL-CSLRT dataset videos")
    parser.add_argument("--skip-face", action="store_true", help="Skip face landmarks extraction")
    parser.add_argument("--data-dir", default="Data/raw", help="Directory containing raw videos")
    parser.add_argument("--output-dir", default="Data/features/mediapipe", help="Directory to save processed features")
    parser.add_argument("--phoenix-dir", default="Data/isl_dataset", help="Directory to save Phoenix format files")
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} does not exist")
        return
    
    # Look for videos in the ISL_CSLRT_Corpus structure
    video_dir = os.path.join(args.data_dir, "ISL_CSLRT_Corpus", "ISL_CSLRT_Corpus", "Videos_Sentence_Level")
    if not os.path.exists(video_dir):
        print(f"Error: Video directory not found at {video_dir}")
        return
        
    print(f"Searching for videos in: {video_dir}")
    
    # Get all sentence directories
    sentence_dirs = []
    sentences = []
    for entry in os.listdir(video_dir):
        full_path = os.path.join(video_dir, entry)
        if os.path.isdir(full_path) and not entry.startswith('.'):
            sentence_dirs.append(full_path)
            # Remove quotes from directory name to get the sentence
            sentence = entry.strip("'")
            sentences.append(sentence)
    
    if not sentence_dirs:
        print(f"Error: No sentence directories found in {video_dir}")
        return
        
    print(f"Found {len(sentence_dirs)} sentence directories to process")
    
    # Process each sentence directory
    print("Processing videos...")
    for sentence_dir, sentence in tqdm(zip(sentence_dirs, sentences), total=len(sentence_dirs)):
        # Look for video files in the sentence directory
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv')
        video_files = []
        for file in os.listdir(sentence_dir):
            if file.lower().endswith(video_extensions):
                video_files.append(os.path.join(sentence_dir, file))
        
        if not video_files:
            print(f"Warning: No video files found in directory: {sentence_dir}")
            continue
            
        # Create output filename from the sentence
        output_name = sentence.replace(' ', '_')
        
        # Process each video in the sentence directory
        for video_path in video_files:
            # Process the video using the sentence name
            result = process_video(video_path, args.output_dir, args.skip_face, output_name=output_name)
            
            if result:
                # Also save the sentence text for later use
                with open(os.path.join(args.output_dir, f"{output_name}.txt"), 'w') as f:
                    f.write(sentence)
                # Only process one video per sentence directory
                break
    
    # Create Phoenix format files
    print("Creating Phoenix format files...")
    create_phoenix_format_files(args.output_dir, args.phoenix_dir)
    
    print("Done!")

if __name__ == "__main__":
    main() 