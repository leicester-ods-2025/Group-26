import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

DATA_PATH = r"C:\Users\vijay\Downloads\tame-pain-trustworthy-assessment-of-pain-from-speech-and-audio-for-the-empowerment-of-patients-1.0.0\tame-pain-trustworthy-assessment-of-pain-from-speech-and-audio-for-the-empowerment-of-patients-1.0.0"

def load_cleaned_metadata(data_path):
    filepath = os.path.join(data_path, "merged_metadata_clean.csv")
    if not os.path.exists(filepath):
        print("merged_metadata_clean.csv not found.")
        print("Run Ashrita's notebook first and save the output.")
        return None
    df = pd.read_csv(filepath)
    print("Loaded successfully.")
    print("Total recordings:", len(df))
    print("Unique participants:", df['PID'].nunique())
    return df

def participant_wise_split(df, participant_col='PID',
                           test_size=0.2, random_state=42):
    participants = df[participant_col].unique()
    print(f"Total participants: {len(participants)}")
    train_pids, test_pids = train_test_split(
        participants,
        test_size=test_size,
        random_state=random_state
    )
    train_df = df[df[participant_col].isin(train_pids)].copy()
    test_df = df[df[participant_col].isin(test_pids)].copy()
    print(f"Training participants: {len(train_pids)}")
    print(f"Testing participants: {len(test_pids)}")
    print(f"Training recordings: {len(train_df)}")
    print(f"Testing recordings: {len(test_df)}")
    print("\nTraining class distribution:")
    print(train_df['pain_category'].value_counts())
    print("\nTesting class distribution:")
    print(test_df['pain_category'].value_counts())
    return train_df, test_df

def save_split_ids(train_df, test_df,
                   data_path, participant_col='PID'):
    train_pids = train_df[[participant_col]].drop_duplicates()
    test_pids = test_df[[participant_col]].drop_duplicates()
    train_path = os.path.join(data_path, "train_pids.csv")
    test_path = os.path.join(data_path, "test_pids.csv")
    train_pids.to_csv(train_path, index=False)
    test_pids.to_csv(test_path, index=False)
    print(f"\nTrain PIDs saved to: {train_path}")
    print(f"Test PIDs saved to: {test_path}")
    print("Share these two files with all group members.")
    print("Do NOT commit to GitHub.")

def load_existing_split(df, data_path, participant_col='PID'):
    train_path = os.path.join(data_path, "train_pids.csv")
    test_path = os.path.join(data_path, "test_pids.csv")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Split files not found. Run save_split_ids first.")
        return None, None
    train_pids = pd.read_csv(train_path)[participant_col].values
    test_pids = pd.read_csv(test_path)[participant_col].values
    train_df = df[df[participant_col].isin(train_pids)].copy()
    test_df = df[df[participant_col].isin(test_pids)].copy()
    print("Split loaded successfully.")
    print(f"Training recordings: {len(train_df)}")
    print(f"Testing recordings: {len(test_df)}")
    return train_df, test_df

if __name__ == "__main__":
    df = load_cleaned_metadata(DATA_PATH)
    if df is not None:
        train_df, test_df = participant_wise_split(df)
        save_split_ids(train_df, test_df, DATA_PATH)
        print("\nDone. Share train_pids.csv and test_pids.csv with the group.")