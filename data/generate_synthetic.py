import os
import pandas as pd
import numpy as np

def generate_synthetic_data(n=1000, file_path='./processed/synthetic_data.csv'):
    # Check if the dataset already exists
    if os.path.exists(file_path):
        # Load existing data
        existing_data = pd.read_csv(file_path)
    else:
        existing_data = pd.DataFrame()

    # Generate new synthetic data
    flashcard_ids = np.arange(1, n+1)
    timestamps = pd.date_range(start="2022-01-01", periods=n, freq="H")
    user_responses = np.random.choice([0, 1], size=n, p=[0.3, 0.7])
    time_taken = np.random.randint(5, 60, size=n)
    flashcard_difficulty = np.random.choice([1, 2, 3, 4, 5], size=n)

    new_data = pd.DataFrame({
        'Flashcard ID': flashcard_ids,
        'Timestamp': timestamps,
        'User Response': user_responses,
        'Time Taken': time_taken,
        'Flashcard Difficulty': flashcard_difficulty
    })

    # Append new data to existing data
    combined_data = pd.concat([existing_data, new_data], ignore_index=True)

    # Save combined data
    combined_data.to_csv(file_path, index=False)

    return combined_data
