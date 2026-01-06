"""
Deep Neural Network with HEAVISIDE activation (Gradient=0)
Refactored for "Non-Contiguous Deep Learning" project
"""
import numpy as np
from typing import Optional, Tuple, List, Dict

class DeepDiscontinuousNeuralNetwork:
    """
    Deep Neural Network with Heaviside activation in ALL hidden layers.
    
    Architecture: input -> hidden1 -> hidden2 -> ... -> hiddenN -> output
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        init_method: str = 'xavier_uniform'
    ):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_hidden_layers = len(hidden_sizes)
        self.output_size = output_size
        self.init_method = init_method
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        # Initialization
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        for i in range(len(layer_sizes) - 1):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i+1]
            
            if init_method == 'xavier_uniform':
                limit = np.sqrt(6.0 / (n_in + n_out))
                W = np.random.uniform(-limit, limit, (n_in, n_out))
            else:
                W = np.random.randn(n_in, n_out) * np.sqrt(1.0 / n_in)
                
            b = np.zeros((1, n_out))
            self.weights.append(W)
            self.biases.append(b)
        
        # Storage
        self.Z_layers = []
        self.A_layers = []
        
        # Adam parameters
        self.m_W = [np.zeros_like(w) for w in self.weights]
        self.v_W = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]
        self.t = 0

    # ------------------------------------------------------------------
    # Activation & Forward
    # ------------------------------------------------------------------
    
    def _heaviside(self, z: np.ndarray) -> np.ndarray:
        """H(x) = 1 if x>=0, 0 else"""
        return np.where(z >= 0, 1.0, 0.0)

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.Z_layers = []
        self.A_layers = [X]
        
        # Hidden layers (Heaviside)
        for i in range(self.num_hidden_layers):
            Z = np.dot(self.A_layers[-1], self.weights[i]) + self.biases[i]
            A = self._heaviside(Z)
            self.Z_layers.append(Z)
            self.A_layers.append(A)
            
        # Output layer (Linear)
        Z_out = np.dot(self.A_layers[-1], self.weights[-1]) + self.biases[-1]
        A_out = Z_out # Linear activation
        
        self.Z_layers.append(Z_out)
        self.A_layers.append(A_out)
        
        return A_out

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    
    def compute_mse(self, Y: np.ndarray, A_out: np.ndarray) -> float:
        return np.mean((Y - A_out) ** 2)

    # ------------------------------------------------------------------
    # Backward & Training (Adam)
    # ------------------------------------------------------------------
    
    def backward_adam(self, X: np.ndarray, Y: np.ndarray, 
                     learning_rate: float = 0.001, 
                     beta1: float = 0.9, beta2: float = 0.999, 
                     epsilon: float = 1e-8) -> float:
        """
        Backpropagation with Adam.
        Returns gradient magnitude (L2 norm of all gradients).
        """
        m = Y.shape[0]
        
        # Output layer gradient (Linear activation, MSE loss)
        # dL/dA_out = 2(A_out - Y)/m
        # dA_out/dZ_out = 1
        dZ = 2 * (self.A_layers[-1] - Y) / m
        
        gradients_W = []
        gradients_b = []
        
        # Output layer gradients
        dW = np.dot(self.A_layers[-2].T, dZ)
        db = np.sum(dZ, axis=0, keepdims=True)
        gradients_W.insert(0, dW)
        gradients_b.insert(0, db)
        
        # Backprop through hidden layers
        dA = np.dot(dZ, self.weights[-1].T)
        
        for i in range(self.num_hidden_layers - 1, -1, -1):
            # Derivative of Heaviside is 0 everywhere (or Dirac at 0, treated as 0 here)
            # This is the expected failure mode
            dZ = dA * 0.0 
            
            dW = np.dot(self.A_layers[i].T, dZ)
            db = np.sum(dZ, axis=0, keepdims=True)
            gradients_W.insert(0, dW)
            gradients_b.insert(0, db)
            
            if i > 0:
                dA = np.dot(dZ, self.weights[i].T)
                
        # Calculate Gradient Magnitude (for reporting)
        total_grad_norm = 0.0
        for g in gradients_W + gradients_b:
            total_grad_norm += np.sum(g ** 2)
        total_grad_norm = np.sqrt(total_grad_norm)

        # Adam Update
        self.t += 1
        for i in range(len(self.weights)):
            # Update W
            self.m_W[i] = beta1 * self.m_W[i] + (1 - beta1) * gradients_W[i]
            self.v_W[i] = beta2 * self.v_W[i] + (1 - beta2) * (gradients_W[i]**2)
            m_hat = self.m_W[i] / (1 - beta1**self.t)
            v_hat = self.v_W[i] / (1 - beta2**self.t)
            self.weights[i] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            
            # Update b
            self.m_b[i] = beta1 * self.m_b[i] + (1 - beta1) * gradients_b[i]
            self.v_b[i] = beta2 * self.v_b[i] + (1 - beta2) * (gradients_b[i]**2)
            m_hat_b = self.m_b[i] / (1 - beta1**self.t)
            v_hat_b = self.v_b[i] / (1 - beta2**self.t)
            self.biases[i] -= learning_rate * m_hat_b / (np.sqrt(v_hat_b) + epsilon)

        return total_grad_norm

    def train(self, X: np.ndarray, Y: np.ndarray, epochs: int, batch_size: int = 32, learning_rate: float = 0.001) -> Dict:
        """Train loop with Batches"""
        history = {'loss': [], 'grad_norm': [], 'zero_grad_pct': 0}
        n_samples = X.shape[0]
        zero_grad_count = 0
        total_updates = 0
        
        for epoch in range(epochs):
            # Shuffle
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]
            
            epoch_loss = 0
            epoch_grad = 0
            n_batches = 0
            
            for start_idx in range(0, n_samples, int(batch_size)):
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                Y_batch = Y_shuffled[start_idx:end_idx]
                
                # Forward
                A_out = self.forward(X_batch)
                loss = self.compute_mse(Y_batch, A_out)
                
                # Backward (Adam)
                grad_norm = self.backward_adam(X_batch, Y_batch, learning_rate=learning_rate)
                
                epoch_loss += loss
                epoch_grad += grad_norm
                n_batches += 1
                
                if grad_norm < 1e-8:
                    zero_grad_count += 1
                total_updates += 1
            
            if n_batches > 0:
                history['loss'].append(epoch_loss / n_batches)
                history['grad_norm'].append(epoch_grad / n_batches)
        
        history['zero_grad_pct'] = (zero_grad_count / total_updates) * 100 if total_updates > 0 else 0
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    # ------------------------------------------------------------------
    # Helper for Vector-based Algos (GWO, Ray)
    # ------------------------------------------------------------------
    def get_total_params(self) -> int:
        total = 0
        for W, b in zip(self.weights, self.biases):
            total += W.size + b.size
        return total
    
    def set_weights_from_vector(self, params: np.ndarray) -> None:
        idx = 0
        for i in range(len(self.weights)):
            W_size = self.weights[i].size
            b_size = self.biases[i].size
            self.weights[i] = params[idx:idx + W_size].reshape(self.weights[i].shape)
            idx += W_size
            self.biases[i] = params[idx:idx + b_size].reshape(self.biases[i].shape)
            idx += b_size
    
    def get_weights_as_vector(self) -> np.ndarray:
        vectors = []
        for W, b in zip(self.weights, self.biases):
            vectors.append(W.flatten())
            vectors.append(b.flatten())
        return np.concatenate(vectors)

def create_network(input_size=2, hidden_sizes=[128, 64, 32], output_size=1):
    return DeepDiscontinuousNeuralNetwork(input_size, hidden_sizes, output_size)
