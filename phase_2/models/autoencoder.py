import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import (
    RANDOM_SEED,
    AE_HIDDEN_DIMS,
    AE_EPOCHS,
    AE_BATCH_SIZE,
    AE_LEARNING_RATE,
    AE_WEIGHT_DECAY,
    AE_VAL_FRACTION,
)


class PacketAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=AE_HIDDEN_DIMS):
        super().__init__()
        encoder_layers = []
        prev = input_dim
        for dim in hidden_dims:
            encoder_layers.extend([nn.Linear(prev, dim), nn.ReLU()])
            prev = dim
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for dim in list(reversed(hidden_dims[:-1])) + [input_dim]:
            decoder_layers.append(nn.Linear(prev, dim))
            if dim != input_dim:
                decoder_layers.append(nn.ReLU())
            prev = dim
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def _benign_train_matrix(x_train, y_train):
    if y_train is not None:
        benign_mask = np.asarray(y_train) == 0
        return x_train.loc[benign_mask].to_numpy(dtype=np.float32)
    return x_train.to_numpy(dtype=np.float32)


def train_autoencoder(x_train, y_train=None, seed=RANDOM_SEED):
    # Fit on benign training rows only; MSE reconstruction loss
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_benign = _benign_train_matrix(x_train, y_train)
    n_val = max(1, int(len(x_benign) * AE_VAL_FRACTION))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(x_benign))
    x_val = x_benign[perm[:n_val]]
    x_fit = x_benign[perm[n_val:]]

    device = torch.device('cpu')
    model = PacketAutoencoder(input_dim=x_fit.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=AE_LEARNING_RATE, weight_decay=AE_WEIGHT_DECAY)
    criterion = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_fit)),
        batch_size=AE_BATCH_SIZE,
        shuffle=True,
    )
    x_val_t = torch.from_numpy(x_val).to(device)

    print(f'Autoencoder training on {len(x_fit)} benign rows ({n_val} val)')
    model.train()
    for epoch in range(1, AE_EPOCHS + 1):
        epoch_loss = 0.0
        n_batches = 0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(x_val_t), x_val_t).item()
        model.train()
        if epoch == 1 or epoch % 5 == 0 or epoch == AE_EPOCHS:
            print(f'  epoch {epoch}/{AE_EPOCHS} train_mse={epoch_loss / n_batches:.6f} val_mse={val_loss:.6f}')

    model.eval()
    return model


def get_anomaly_scores(model, x_data, batch_size=AE_BATCH_SIZE):
    # Mean squared reconstruction error per row (higher = more anomalous)
    model.eval()
    x_np = np.array(
        x_data.to_numpy(dtype=np.float32) if hasattr(x_data, 'to_numpy') else x_data,
        dtype=np.float32,
        copy=True,
    )
    loader = DataLoader(TensorDataset(torch.from_numpy(x_np)), batch_size=batch_size, shuffle=False)
    errors = []
    with torch.no_grad():
        for (batch,) in loader:
            recon = model(batch)
            mse = torch.mean((recon - batch) ** 2, dim=1)
            errors.append(mse.numpy())
    return np.concatenate(errors)
