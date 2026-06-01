import pickle
import os

def save_model_artifacts(model, mean, std, filename="rnn_artifacts.pkl"):
    """
    Serializes and saves the trained weights, biases, and normalization parameters.
    """
    artifacts = {
        'W_xh': model.W_xh,
        'W_hh': model.W_hh,
        'W_hy': model.W_hy,
        'b_h': model.b_h,
        'b_y': model.b_y,
        'mean': mean,
        'std': std,
        'hidden_dim': model.hidden_dim
    }

    filepath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(artifacts, f)
    print(f"[System Execution] Artifacts safely serialized and saved to: {filepath}")

def load_model_artifacts(filename="rnn_artifacts.pkl"):
    """
    Loads model weights and normalization constants for inference.
    """
    filepath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', filename)
    with open(filepath, 'rb') as f:
        artifacts = pickle.load(f)
    print(f"[System Execution] Artifacts loaded successfully from: {filepath}")
    return artifacts