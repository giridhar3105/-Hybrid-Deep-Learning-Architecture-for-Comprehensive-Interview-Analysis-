import pickle
import os

def save_model(model, file_path):
    """Save the trained model to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        print(f"Model saved to {file_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

def load_model(file_path):
    """Load a trained model from a file."""
    if os.path.exists(file_path):
        try:
            with open(file_path, 'rb') as file:
                model = pickle.load(file)
            print(f"Model loaded from {file_path}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"No model found at {file_path}. Starting fresh.")
    return None
