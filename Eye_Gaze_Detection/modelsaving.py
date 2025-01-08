import pickle

def save_model(model, filename):
    """Save a trained model to disk."""
    with open(filename, 'wb') as file:
        pickle.dump(model, file)


def load_model(filename):
    """Load a saved model from disk."""
    with open(filename, 'rb') as file:
        return pickle.load(file)