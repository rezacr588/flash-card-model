from src.preprocess_data import load_data, preprocess_data, split_data
from src.train_model import train_and_evaluate
from src.save_load_model import save_model
from data.generate_synthetic import generate_synthetic_data

def main():
    # Generate synthetic data
    generate_synthetic_data()

    # Load and preprocess the data
    data = load_data()
    data = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(data)

    # Train the model and evaluate its accuracy
    model, accuracy = train_and_evaluate(X_train, y_train, X_test, y_test)
    print(f"Enhanced Model Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()
