import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data():
    data = pd.read_csv('processed/synthetic_data.csv')
    return data

def preprocess_data(data):
    # Handle Missing Values
    data.dropna(inplace=True)
    
    # Extract features from 'Timestamp'
    data['Hour'] = pd.to_datetime(data['Timestamp']).dt.hour
    data['Day'] = pd.to_datetime(data['Timestamp']).dt.day
    data['Day_of_Week'] = pd.to_datetime(data['Timestamp']).dt.dayofweek
    
    # One-hot encoding for 'Flashcard Difficulty'
    data = pd.get_dummies(data, columns=['Flashcard Difficulty'], prefix='Difficulty')
    
    # Create 'Hour_Difficulty' interaction feature
    if 'Difficulty_Medium' in data.columns:
        data['Hour_Difficulty'] = data['Hour'] * data['Difficulty_Medium']
    else:
        data['Hour_Difficulty'] = 0
    
    # Normalize 'Time Taken' feature
    scaler = StandardScaler()
    data['Time Taken'] = scaler.fit_transform(data[['Time Taken']])
    
    # Drop the 'Timestamp' column
    data.drop('Timestamp', axis=1, inplace=True)
    
    return data

def split_data(data):
    features = data.drop('User Response', axis=1)
    target = data['User Response']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
