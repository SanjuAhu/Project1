import numpy as np

class RNN:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.001):
        # Xavier/Glorot weight initialization for stable training
        self.W_xh = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + hidden_dim))
        self.W_hy = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        
        self.b_h = np.zeros((hidden_dim, 1))
        self.b_y = np.zeros((output_dim, 1))
        
        self.hidden_dim = hidden_dim

    def forward_propagation(self, X_batch):
        m, T, n_x = X_batch.shape
        hidden_states = np.zeros((T, self.hidden_dim, m))
        h_current = np.zeros((self.hidden_dim, m))
        
        for t in range(T):
            X_t = X_batch[:, t, :].T  # Shape: (n_x, m)
            Z = np.dot(self.W_xh, X_t) + np.dot(self.W_hh, h_current) + self.b_h
            h_current = np.tanh(Z)
            hidden_states[t] = h_current
            
        y_pred = np.dot(self.W_hy, h_current) + self.b_y  # Final time step output
        return y_pred.T, hidden_states

    def backward_propagation(self, X_batch, y_batch, y_pred, hidden_states):
        m, T, n_x = X_batch.shape
        
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)
        
        # Mean Squared Error derivative: dJ/dy_pred = (y_pred - y_true) / m
        dy = (y_pred - y_batch.reshape(-1, 1)).T  # Shape: (1, m)
        
        # Output weights gradient calculation from final step T-1
        h_T = hidden_states[T-1]
        dW_hy += np.dot(dy, h_T.T)
        db_y += np.sum(dy, axis=1, keepdims=True)
        
        # Initialize the backpropagation vector through the hidden layer
        dh_next = np.dot(self.W_hy.T, dy)  # Shape: (hidden_dim, m)
        
        # --- THE RECURSIVE TEMPORAL LOOP ---
        for t in reversed(range(T)):
            h_current = hidden_states[t]
            
            # Gradients through the non-linear Tanh barrier
            dtanh = (1 - h_current ** 2) * dh_next  # Shape: (hidden_dim, m)
            
            X_t = X_batch[:, t, :].T  # Shape: (n_x, m)
            h_prev = hidden_states[t-1] if t > 0 else np.zeros_like(h_current)
            
            # Accumulating parameter changes over time
            dW_xh += np.dot(dtanh, X_t.T)
            dW_hh += np.dot(dtanh, h_prev.T)
            db_h += np.sum(dtanh, axis=1, keepdims=True)
            
            # Propagate the error vector back to the previous time step
            dh_next = np.dot(self.W_hh.T, dtanh)

            # --- GRADIENT CLIPPING ---
            # Capping the stride speed to avoid explosion errors
            max_norm = 5.0
            for grad in [dW_xh, dW_hh, dW_hy, db_h, db_y]:
                norm = np.sqrt[np.sum(grad ** 2)]
                if norm > max_norm:
                    grad *= (max_norm / norm)
            
        return dW_xh, dW_hh, dW_hy, db_h, db_y
    