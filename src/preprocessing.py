import librosa
import numpy as np
import pandas as pd
import os

# 1. Main Directory containing the generated clean CSV file
DATA_PATH = r"C:\Users\alekh\Downloads\tame-pain-trustworthy-assessment-of-pain-from-speech-and-audio-for-the-empowerment-of-patients-1.0.0\tame-pain-trustworthy-assessment-of-pain-from-speech-and-audio-for-the-empowerment-of-patients-1.0.0"

# 2. Extracted audio directory location verified via command line
AUDIO_PATH = r"C:\Users\alekh\Downloads\mic1_trim_v2"

def extract_features(file_path, sr=16000):
    try:
        y, sr = librosa.load(file_path, sr=sr)
        if len(y) < sr * 0.5:
            return None
            
        features = {}
        
        # MFCC Features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
            
        # Pitch Tracking
        pitches, _ = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[pitches > 0]
        features['pitch_mean'] = np.mean(pitch_values) if len(pitch_values) > 0 else 0
        features['pitch_std'] = np.std(pitch_values) if len(pitch_values) > 0 else 0
        
        # Energy / RMS
        rms = librosa.feature.rms(y=y)
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # Spectral Centroid
        sc = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['spectral_centroid_mean'] = np.mean(sc)
        features['spectral_centroid_std'] = np.std(sc)
        
        # Spectral Bandwidth
        sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features['spectral_bandwidth_mean'] = np.mean(sb)
        features['spectral_bandwidth_std'] = np.std(sb)
        
        # Spectral Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features['spectral_rolloff_mean'] = np.mean(rolloff)
        
        # Speaking Rate Calculation
        frames = librosa.effects.split(y, top_db=20)
        total_voiced = sum(end - start for start, end in frames)
        duration = len(y) / sr
        features['speaking_rate'] = total_voiced / (sr * duration) if duration > 0 else 0
        
        # Jitter Calculation (With zero-division protection)
        if len(pitch_values) > 1:
            periods = 1.0 / (pitch_values + 1e-6)
            mean_period = np.mean(periods)
            if mean_period > 0:
                features['jitter'] = np.mean(np.abs(np.diff(periods))) / mean_period
            else:
                features['jitter'] = 0
        else:
            features['jitter'] = 0
            
        # Shimmer Calculation
        amplitude = np.abs(y)
        if len(amplitude) > 1:
            features['shimmer'] = np.mean(np.abs(np.diff(amplitude))) / (np.mean(amplitude) + 1e-6)
        else:
            features['shimmer'] = 0
            
        return features
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def build_feature_matrix(metadata_df, audio_path):
    all_features = []
    total = len(metadata_df)
    print(f"Extracting features from {total} recordings...")
    
    for idx, row in metadata_df.iterrows():
        pid = row['PID']
        uttid = row['UTTID']
        cond = row['COND']
        uttnum = row['UTTNUM']
        
        filename = f"{pid}.{cond}.{uttnum}.{uttid}.wav"
        file_path = os.path.join(audio_path, pid, filename)
        
        if os.path.exists(file_path):
            features = extract_features(file_path)
            if features is not None:
                features['PID'] = pid
                features['filename'] = filename
                features['pain_category'] = row['pain_category']
                features['PAIN LEVEL'] = row['PAIN LEVEL']
                features['ACTION LABEL'] = row['ACTION LABEL']
                all_features.append(features)
        else:
            # Troubleshooting warning for the first broken file path check
            if idx == 0:
                print(f"DEBUG Path Warning - Check layout here: {file_path}")
            
        if (idx + 1) % 200 == 0:
            print(f"Processed {idx + 1} of {total}")
            
    feature_df = pd.DataFrame(all_features)
    print(f"Done. Total features extracted: {len(feature_df)}")
    return feature_df

def save_feature_matrix(feature_df, data_path):
    output_path = os.path.join(data_path, "feature_matrix.csv")
    feature_df.to_csv(output_path, index=False)
    print(f"Successfully saved features to: {output_path}")

if __name__ == "__main__":
    metadata_path = os.path.join(DATA_PATH, "merged_metadata_clean.csv")
    
    if not os.path.exists(metadata_path):
        print(f"File missing at: {metadata_path}")
        print("Please run Ashrita's preprocessing script to generate it first.")
    else:
        metadata_df = pd.read_csv(metadata_path)
        feature_df = build_feature_matrix(metadata_df, AUDIO_PATH)
        if not feature_df.empty:
            save_feature_matrix(feature_df, DATA_PATH)