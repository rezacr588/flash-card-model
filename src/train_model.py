from joblib import load, dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import os
from joblib import load, dump

def train_and_evaluate(X_train, y_train, X_test, y_test, model_path='trained_model.joblib'):
    # Check if a previously trained model exists
    if os.path.exists(model_path):
        # Load the existing model
        model = load(model_path)
        print("Loaded existing model.")
    else:
        # Initialize a new Random Forest Classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        print("Created new model.")

    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    
    # Print classification report for more detailed metrics
    print(classification_report(y_test, predictions))
    
    # Save the updated model
    dump(model, model_path)
    
    return model, accuracy
