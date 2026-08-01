import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from ..config import (
        RANDOM_SEED,
        AE_HIDDEN_DIMS,
        AE_HIDDEN_DIMS_GRID,
        AE_EPOCHS,
        AE_BATCH_SIZE,
        AE_LEARNING_RATE,
        AE_WEIGHT_DECAY,
        AE_VAL_FRACTION,
    )
except ImportError:  # Preserve direct imports used by phase_2/main.py.
    from config import (
        RANDOM_SEED,
        AE_HIDDEN_DIMS,
        AE_HIDDEN_DIMS_GRID,
        AE_EPOCHS,
        AE_BATCH_SIZE,
        AE_LEARNING_RATE,
        AE_WEIGHT_DECAY,
        AE_VAL_FRACTION,
    )


class PacketAutoencoder(nn.Module):
    """Fully connected autoencoder for packet-level behaviour features."""

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
        """Reconstruct a batch of packet features."""
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x):
        """Return the bottleneck representation for a batch."""
        return self.encoder(x)


def _benign_train_matrix(x_train, y_train):
    if y_train is not None:
        benign_mask = np.asarray(y_train) == 0
        return x_train.loc[benign_mask].to_numpy(dtype=np.float32)
    return x_train.to_numpy(dtype=np.float32)


def _benign_fit_val_split(x_train, y_train, seed, val_fraction=AE_VAL_FRACTION):
    # Same split for every tuning config (fixed seed) so val MSE is comparable
    x_benign = _benign_train_matrix(x_train, y_train)
    if len(x_benign) < 2:
        raise ValueError("Autoencoder training requires at least two benign rows.")
    n_val = min(len(x_benign) - 1, max(1, int(len(x_benign) * val_fraction)))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(x_benign))
    x_val = x_benign[perm[:n_val]]
    x_fit = x_benign[perm[n_val:]]
    return x_fit, x_val


def _fit_autoencoder(
    x_fit,
    x_val,
    hidden_dims,
    seed,
    verbose=True,
    epochs=AE_EPOCHS,
    batch_size=AE_BATCH_SIZE,
    learning_rate=AE_LEARNING_RATE,
    weight_decay=AE_WEIGHT_DECAY,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cpu')
    model = PacketAutoencoder(input_dim=x_fit.shape[1], hidden_dims=hidden_dims).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_fit)),
        batch_size=batch_size,
        shuffle=True,
    )
    x_val_t = torch.from_numpy(x_val).to(device)

    val_loss = float('nan')
    model.train()
    for epoch in range(1, epochs + 1):
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
        if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == epochs):
            print(f'  epoch {epoch}/{epochs} train_mse={epoch_loss / n_batches:.6f} val_mse={val_loss:.6f}')

    model.eval()
    return model, val_loss


def train_autoencoder(
    x_train,
    y_train=None,
    hidden_dims=AE_HIDDEN_DIMS,
    seed=RANDOM_SEED,
    *,
    epochs=AE_EPOCHS,
    batch_size=AE_BATCH_SIZE,
    learning_rate=AE_LEARNING_RATE,
    weight_decay=AE_WEIGHT_DECAY,
    val_fraction=AE_VAL_FRACTION,
    verbose=True,
):
    """Fit an autoencoder on benign training rows only."""
    x_fit, x_val = _benign_fit_val_split(
        x_train, y_train, seed, val_fraction=val_fraction
    )
    print(f'Autoencoder training on {len(x_fit)} benign rows ({len(x_val)} val), dims={hidden_dims}')
    model, _ = _fit_autoencoder(
        x_fit,
        x_val,
        hidden_dims,
        seed,
        verbose=verbose,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    return model


def tune_autoencoder(x_train, y_train=None, grid=AE_HIDDEN_DIMS_GRID, seed=RANDOM_SEED):
    """Select an architecture using benign validation reconstruction MSE."""
    x_fit, x_val = _benign_fit_val_split(x_train, y_train, seed)
    print(f'Tuning autoencoder over {len(grid)} architectures on {len(x_fit)} benign rows ({len(x_val)} val)')

    results = []
    best_model = None
    best_dims = None
    best_val = float('inf')
    for hidden_dims in grid:
        print(f'--- config dims={hidden_dims} ---')
        model, val_loss = _fit_autoencoder(x_fit, x_val, hidden_dims, seed)
        results.append((hidden_dims, val_loss))
        print(f'  -> val_mse={val_loss:.6f}')
        if val_loss < best_val:
            best_val = val_loss
            best_dims = hidden_dims
            best_model = model

    print(f'Best AE architecture: {best_dims} (val_mse={best_val:.6f})')
    return best_model, best_dims, results


def get_anomaly_scores(model, x_data, batch_size=AE_BATCH_SIZE):
    """Return per-row reconstruction MSE; higher means more anomalous."""
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


def get_latent_embedding(model, x_data, batch_size=AE_BATCH_SIZE):
    """Return the encoder bottleneck representation for every row."""
    model.eval()
    x_np = np.array(
        x_data.to_numpy(dtype=np.float32) if hasattr(x_data, 'to_numpy') else x_data,
        dtype=np.float32,
        copy=True,
    )
    loader = DataLoader(TensorDataset(torch.from_numpy(x_np)), batch_size=batch_size, shuffle=False)
    latents = []
    with torch.no_grad():
        for (batch,) in loader:
            latents.append(model.encode(batch).numpy())
    return np.concatenate(latents)
