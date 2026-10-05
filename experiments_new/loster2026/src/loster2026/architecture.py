"""Active legacy net/model.py only: ResidualBlock, PredictionResidualBlock, AE.
Audit 1 F08/F27/F28: no RevIN, global skip, shared weights or dormant centers.
"""
import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.0, norm_eps=1e-5):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(),
                                   nn.Linear(hidden_size, output_size), nn.Dropout(dropout))
        self.residual = nn.Linear(input_size, output_size)
        self.layer_norm = nn.LayerNorm(output_size, eps=norm_eps)

    def forward(self, x):
        return self.layer_norm(self.block(x) + self.residual(x))

class PredictionResidualBlock(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(),
                                   nn.Linear(hidden_size, output_size))
        self.residual = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.block(x) + self.residual(x)

class Autoencoder(nn.Module):
    def __init__(self, length, config):
        super().__init__()
        d = config.latent_dim
        self.length = length
        self.dense_encoder = nn.Sequential(*[
            ResidualBlock(length if i == 0 else d, d, d, config.dropout, config.layer_norm_eps)
            for i in range(config.encoder_blocks)])
        self.dense_decoder = nn.Sequential(*[
            PredictionResidualBlock(d, d, length) if i == config.decoder_blocks - 1
            else ResidualBlock(d, d, d, config.dropout, config.layer_norm_eps)
            for i in range(config.decoder_blocks)])

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (self.length, 1):
            raise ValueError("Expected [B,L,1] fixed-length univariate input")
        z = self.dense_encoder(x.squeeze(2))
        return self.dense_decoder(z).unsqueeze(2), z

class TwoViewModel(nn.Module):
    def __init__(self, length, clusters, config):
        super().__init__()
        self.original = Autoencoder(length, config)
        self.augmented = Autoencoder(length, config)
        self.centers_original = nn.Parameter(torch.zeros(clusters, config.latent_dim))
        self.centers_augmented = nn.Parameter(torch.zeros(clusters, config.latent_dim))
