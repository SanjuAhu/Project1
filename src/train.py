import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_pipeline import load_and_preprocess
from rnn import RNN

def train_network(file_path, epochs=100, batch_size=64, hidden_dim=32, lr=0.0005):
    print("[System Notification] Loading NASA Turbofan Telemetry Data...")
    X, y, mean, std = load_and_preprocess(file_path, sequence_length=30)

    num_samples, seq_length, input_dim = X.shape
    print(f"[System Status] Tensor Loaded successfully. Samples: {num_samples}, Time Steps: {seq_length}, Sensors: {input_dim}")

    model = RNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=1, learning_rate=lr)
    print(f"[Initialization] Version 23.0 Training Engine active. Optimization starting...")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        permutation = np.random.permutation(num_samples)
        X_shuffled = X[permutation]
        y_shuffled = y[permutation]

        epoch_loss = 0
        num_batches = int(np.ceil(num_samples / batch_size))

        for b in range(num_batches):
            start_idx = b * batch_size
            end_idx = min(start_idx + batch_size, num_samples)

            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]

            # 1. Forward Pass
            y_pred, hidden_states = model.forward_propagation(X_batch)

            # 2. Compute Mean Squared Error Loss for tracking
            batch_loss = np.mean((y_pred - y_batch.reshape(-1, 1)) ** 2) / 2.0
            epoch_loss += batch_loss * (end_idx - start_idx)

            # 3. Backward Pass (Compute Gradients)
            dW_xh, dW_hh, dW_hy, db_h, db_y = model.backward_propagation(
                X_batch, y_batch, y_pred, hidden_states
            )

            # 4. Parameter Optimization Update Step (SGD)
            model.W_xh -= model.lr * dW_xh
            model.W_hh -= model.lr * dW_hh
            model.W_hy -= model.lr * dW_hy
            model.b_h -= model.lr * db_h
            model.b_y -= model.lr * db_y
        
        total_epoch_loss = epoch_loss / num_samples

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:03d}/{epochs:03d} | Total Normalized MSE Loss: {total_epoch_loss:.4f}")
        
    print("-" * 60)
    print(f"Samples: {num_samples}")
    print("[System Complete] Model trained successfully. Weights optimized.")
    return model, mean, std

if __name__ == "__main__":
    # Path configuration pointing to your raw NASA text files
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw', 'train_FD001.txt')
    
    # Run the compiled optimization engine
    trained_model, data_mean, data_std = train_network(data_path, epochs=100, batch_size=32, hidden_dim=16, lr=0.000075)

    from utils import save_model_artifacts

    save_model_artifacts(trained_model, data_mean, data_std)