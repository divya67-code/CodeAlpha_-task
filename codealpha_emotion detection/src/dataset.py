import os
import glob
import numpy as np
import pandas as pd
from features import extract_all_features
from preprocess import preprocess_pipeline

def parse_ravdess_labels(filepath):
    """
    Parse emotion label from RAVDESS filename.
    Filename example: 03-01-01-01-01-01-01.wav
    Emotion is the 3rd identifier (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
    """
    filename = os.path.basename(filepath)
    parts = filename.split('-')
    if len(parts) >= 3:
        emotion_map = {
            '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad', 
            '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
        }
        return emotion_map.get(parts[2], 'unknown')
    return 'unknown'

def create_dataset_dataframe(data_dir):
    """
    Walk through the dataset directory, find audio files, and extract labels.
    Currently configured specifically for RAVDESS as a baseline.
    Modify or extend for TESS and EMO-DB.
    """
    file_paths = []
    labels = []
    
    # Assuming standard RAVDESS structure where standard files are .wav
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                label = parse_ravdess_labels(file)
                if label != 'unknown':
                    file_paths.append(file_path)
                    labels.append(label)
                    
    df = pd.DataFrame({'filepath': file_paths, 'emotion': labels})
    return df

def feature_generator(dataframe, augment=False):
    """
    A generator to yield extracted features and labels sequentially.
    Ideal for large datasets that don't fit into memory.
    """
    for index, row in dataframe.iterrows():
        try:
            audio, sr = preprocess_pipeline(row['filepath'], augment=augment)
            features = extract_all_features(audio, sr)
            yield features, row['emotion']
        except Exception as e:
            print(f"Error processing {row['filepath']}: {e}")

def load_data_to_memory(dataframe, augment=False):
    """
    Load all features and labels into memory for smaller datasets.
    """
    X = []
    y = []
    for index, row in dataframe.iterrows():
        try:
            audio, sr = preprocess_pipeline(row['filepath'], augment=augment)
            features = extract_all_features(audio, sr)
            X.append(features)
            y.append(row['emotion'])
        except Exception as e:
            print(f"Error processing {row['filepath']}: {e}")
            
    return np.array(X), np.array(y)

if __name__ == "__main__":
    import sys
    print("Testing dataset.py...")
    
    # Check data directory relative to the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    
    print(f"Searching for audio files in: {data_dir}")
    df = create_dataset_dataframe(data_dir)
    
    print(f"Found {len(df)} files.")
    if len(df) > 0:
        print("First few entries in the dataset:")
        print(df.head())
    else:
        print("No audio files found. Please make sure dataset WAV files are placed in the 'data' directory.")
        
    print("dataset.py script executed successfully.")
