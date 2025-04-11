#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to download and prepare the ISL-CSLRT dataset for sign language translation
"""
import os
import time
import subprocess
import shutil
from pathlib import Path

# Create needed directories
DATA_DIR = Path('Data')
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
FEATURES_DIR = DATA_DIR / 'features/mediapipe'
ISL_DATASET_DIR = DATA_DIR / 'isl_dataset'

for dir_path in [DATA_DIR, RAW_DIR, PROCESSED_DIR, FEATURES_DIR, ISL_DATASET_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

def download_with_retry():
    """Download the dataset with retry logic"""
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            print(f"Download attempt {attempt+1}/{max_retries}...")
            
            # Try to use kagglehub first (more reliable)
            try:
                import kagglehub
                print("Using kagglehub to download the dataset...")
                path = kagglehub.dataset_download("drblack00/isl-csltr-indian-sign-language-dataset")
                print(f"Dataset downloaded to: {path}")
                
                # Copy contents to project's raw data directory
                for item in Path(path).glob('*'):
                    dest_path = RAW_DIR / item.name
                    if item.is_dir():
                        shutil.copytree(item, dest_path, dirs_exist_ok=True)
                    else:
                        shutil.copy2(item, dest_path)
                
                print("Dataset copied to project directory.")
                return True
            except Exception as e:
                print(f"kagglehub download failed: {e}")
                print("Falling back to kaggle CLI...")
            
            # Try kaggle CLI as fallback
            dataset_slug = 'drblack00/isl-csltr-indian-sign-language-dataset'
            cmd = f"kaggle datasets download {dataset_slug} -p {RAW_DIR} --unzip"
            
            # Run with timeout
            process = subprocess.run(
                cmd, 
                shell=True,
                timeout=600  # 10 minute timeout
            )
            
            if process.returncode == 0:
                print("Download completed successfully!")
                return True
                
        except subprocess.TimeoutExpired:
            print(f"Download timed out on attempt {attempt+1}")
        except Exception as e:
            print(f"Error during download: {e}")
        
        print(f"Retrying in {retry_delay} seconds...")
        time.sleep(retry_delay)
    
    return False

def verify_download():
    """Verify the dataset was downloaded correctly"""
    corpus_dir = RAW_DIR / 'ISL_CSLRT_Corpus'
    if not corpus_dir.exists():
        potential_paths = list(RAW_DIR.glob('*'))
        for path in potential_paths:
            if 'ISL' in path.name and path.is_dir():
                corpus_dir = path
                break
    
    if not corpus_dir.exists():
        print("Could not find ISL_CSLRT_Corpus directory")
        return False
        
    # Check for key directories
    word_frames_dir = corpus_dir / 'Frames_Word_Level'
    sentence_frames_dir = corpus_dir / 'Frames_Sentence_Level'
    metadata_dir = corpus_dir / 'corpus_csv_files'
    
    missing_dirs = []
    for dir_path, dir_name in [(word_frames_dir, "Word Frames"), 
                               (sentence_frames_dir, "Sentence Frames"),
                               (metadata_dir, "Metadata CSV Files")]:
        if not dir_path.exists():
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"Missing directories: {', '.join(missing_dirs)}")
        return False
        
    return True

if __name__ == "__main__":
    print("Downloading ISL-CSLRT dataset...")
    success = download_with_retry()
    
    if success and verify_download():
        print("Dataset downloaded successfully.")
        print("Now run process_isl_dataset.py to process the data.")
    else:
        print("Failed to download or verify the dataset.")
        print("Try manually downloading from: https://www.kaggle.com/datasets/drblack00/isl-csltr-indian-sign-language-dataset")
        print("After downloading, place the unzipped files in Data/raw/ directory.") 