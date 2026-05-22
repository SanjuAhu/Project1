import numpy as np

class RNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        
        # Initialize weights
        self.W_xh = np.random.randn(hidden_dim, input_dim) * 0.01
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.W_hy = np.random.randn(output_dim, hidden_dim) * 0.01

        self.b_h = np.zeros((hidden_dim, 1))
        self.b_y = np.zeros((output_dim, 1))

        self.hidden_dim = hidden_dim

    def forward_propagation(self, X_batch):
        m, T, x_n = X_batch.shape

        hidden_states = np.zeros((T, self.hidden_dim, m))

        h_current = np.zeros((self.hidden_dim, m))

        for t in range(T):
            X_t = X_batch[:, t, :].T

            Z = np.dot(self.W_xh, X_t) + np.dot(self.W_hh, h_current) + self.b_h
            h_current = np.tanh(Z)

            hidden_states[t] = h_current
        
        y_pred = np.dot(self.W_hy, h_current) + self.b_y

        return y_pred.T, hidden_states
    



rnn = RNN(input_dim=24, hidden_dim=32, output_dim=1)
X_dummy = np.random.randn(8, 30, 24)
y_pred, hidden_states = rnn.forward_propagation(X_dummy)

print(y_pred.shape)
print(hidden_states.shape)

