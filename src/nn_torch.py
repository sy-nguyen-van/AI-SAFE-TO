import torch
import torch.nn as nn

# =============================================================================
# Fourier Feature Expansion
# =============================================================================
class FourierFeatureTransform(nn.Module):
    """
    Gaussian Fourier feature mapping.
    This embeds raw coordinates x into a high-frequency sinusoidal space.

    Input:
        x ∈ ℝ^(N×input_dim)
    Output:
        y ∈ ℝ^(N×2*mapping_size)  →  [sin(Bx), cos(Bx)]

    Benefit:
        Allows the neural network to represent fine structural details
        (useful for high-resolution density fields in topology optimization).
    """
    def __init__(self, input_dim, mapping_size=256, scale=10.0):
        super().__init__()
        # Random Gaussian matrix B (mapping_size × input_dim), frozen (no gradient)
        self.B = nn.Parameter(torch.randn(mapping_size, input_dim) * scale, requires_grad=False)

    def forward(self, x):
        # Project inputs into Fourier basis: (2πx)B^T
        x_proj = (2.0 * torch.pi * x) @ self.B.T
        # Concatenate sin and cos embeddings
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


# =============================================================================
# Positional Encoding (NeRF-style)
# =============================================================================
class PositionalEncoding(nn.Module):
    """
    Deterministic Positional Encoding (sin/cos of 2^k * pi * x).
    Used in original NeRF and Transformers.
    """
    def __init__(self, input_dim, num_freqs=10):
        super().__init__()
        self.input_dim = input_dim
        self.num_freqs = num_freqs
        # Frequencies: 2^0, 2^1, ... 2^(L-1)
        self.register_buffer("freq_bands", 2.0 ** torch.linspace(0., num_freqs - 1, num_freqs))

    def forward(self, x):
        # x: (N, input_dim)
        embeds = []
        # We can append raw x if desired, but here we just do freq bands
        for freq in self.freq_bands:
            embeds.append(torch.sin(torch.pi * x * freq))
            embeds.append(torch.cos(torch.pi * x * freq))
        # Concatenate: (N, input_dim * 2 * num_freqs)
        return torch.cat(embeds, dim=-1)


# =============================================================================
# Custom Sine Activation
# =============================================================================
class SirenActivation(nn.Module):
    """
    SIREN-style activation (sin(x)):
    Enables smoother representation of PDE-like fields, good for physical fields.
    """
    def forward(self, x):
        return torch.sin(x)


# =============================================================================
# Density Prediction Neural Network
# =============================================================================
class DensityNetwork(nn.Module):
    """
    Neural network mapping (x,y) -> density(x,y) in [0,1].

    Architecture:
        Optional Fourier/Positional feature embedding
        -> (Optional Gating Network)
        -> Fully-connected layers
        -> 1 output neuron + Sigmoid activation
    """
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=4,
                 activation='relu', use_fourier=False, fourier_scale=1.0,
                 feature_type='fourier'):
        super().__init__()


        
        # -------- Optional Feature Embedding --------
        if use_fourier:
            if feature_type == 'positional':
                # Deterministic Positional Encoding
                num_freqs = int(fourier_scale) if fourier_scale > 1.0 else 6
                self.embedder = PositionalEncoding(input_dim, num_freqs=num_freqs)
                current_dim = input_dim * 2 * num_freqs
            
            else:
                # Default: Gaussian Fourier Features
                f_dim = 128
                self.embedder = FourierFeatureTransform(input_dim, f_dim, fourier_scale)
                current_dim = f_dim * 2
        else:
            self.embedder = None
            current_dim = input_dim  # directly feed raw (x,y)
        


        
        # -------- Input Layer --------
        layers = []
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(self._get_activation(activation))

        # -------- Hidden Layers --------
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(self._get_activation(activation))

        # -------- Output Layer --------
        layers.append(nn.Linear(hidden_dim, 1))
        # Sigmoid ensures density value in [0,1]
        layers.append(nn.Sigmoid())

        # Combine all into sequential model
        self.net = nn.Sequential(*layers)

        # Initialize weights properly based on activation type
        self._init_weights(activation)

    # -------------------------------------------------------------------------
    # Activation selection helper
    # -------------------------------------------------------------------------
    def _get_activation(self, name):
        """Return activation layer based on keyword."""
        name = name.lower()
        if name == 'tanh':
            return nn.Tanh()
        if name == 'relu':
            return nn.ReLU()
        if name == 'leaky_relu':
            return nn.LeakyReLU(0.2)
        if name == 'siren':
            return SirenActivation()
        # default activation
        return nn.ReLU()

    # -------------------------------------------------------------------------
    # Weight Initialization
    # -------------------------------------------------------------------------
    def _init_weights(self, act):
        """
        Correct initialization improves convergence.

        ReLU → Kaiming initialization
        SIREN → Uniform init for sinusoidal networks
        """
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                if act.lower() == 'siren':
                    # Recommended initialization for SIREN
                    nn.init.uniform_(m.weight, -1.0, 1.0)
                elif act.lower() in ['tanh', 'sigmoid']:
                    # Xavier (Glorot) for Tanh/Sigmoid
                    nn.init.xavier_uniform_(m.weight)
                else:
                    # Standard for ReLU networks
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    # -------------------------------------------------------------------------
    # Forward Pass
    # -------------------------------------------------------------------------
    def forward(self, x):
        """
        Forward propagation:
            x: (N, input_dim) -> coordinates (e.g., FE mesh nodes)

        Returns:
            density values (N*1) in [0,1]
        """


        # Apply Feature Embedding if used
        if self.embedder is not None:
            x = self.embedder(x)

        # Pass through neural network
        return self.net(x)