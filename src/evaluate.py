import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rnn import RNN
from data_pipeline import load_and_preprocess
from utils import load_model_artifacts

def evaluate_on_test_set(test_data_path, rul_data_path):
    print("[Validation Engine] Loading saved model parameters...")
    artifacts = load_model_artifacts()
    
    # Reconstruct architecture from saved state
    model = RNN(input_dim=artifacts['mean'].shape[0], hidden_dim=artifacts['hidden_dim'], output_dim=1)
    model.W_xh, model.W_hh, model.W_hy = artifacts['W_xh'], artifacts['W_hh'], artifacts['W_hy']
    model.b_h, model.b_y = artifacts['b_h'], artifacts['b_y']
    
    # Call unified pipeline using structural consistency
    X_test, test_engines, _, _ = load_and_preprocess(
        test_data_path, sequence_length=30, mode='test', 
        mean=artifacts['mean'], std=artifacts['std']
    )
    
    true_rul = np.loadtxt(rul_data_path)
    predictions = []
    ground_truth = []
    
    print("[Validation Engine] Processing optimized data pipeline...")
    for idx, engine in enumerate(test_engines):
        # Array indices are 0-based; engine IDs are 1-based
        engine_idx = int(engine) - 1 
        
        X_tensor = np.expand_dims(X_test[idx], axis=0)
        y_pred, _ = model.forward_propagation(X_tensor)
        
        predictions.append(y_pred[0, 0])
        ground_truth.append(true_rul[engine_idx])
        
    predictions, ground_truth = np.array(predictions), np.array(ground_truth)
    
    mae = np.mean(np.abs(predictions - ground_truth))
    rmse = np.sqrt(np.mean((predictions - ground_truth) ** 2))
    
    print("-" * 60)
    print(f"[Evaluation Complete] Clean Structural Test MAE: {mae:.2f} Flight Cycles")
    print(f"[Evaluation Complete] Clean Structural Test RMSE: {rmse:.2f} Flight Cycles")
    print("-" * 60)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    evaluate_on_test_set(
        os.path.join(base_dir, 'data', 'raw', 'test_FD001.txt'),
        os.path.join(base_dir, 'data', 'raw', 'RUL_FD001.txt')
    )