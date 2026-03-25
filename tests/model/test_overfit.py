from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import HOVDatasetConfig, HOVEncoderDecoderDataset
from src.model import DequantTransformer, DequantTransformerConfig


def test_dequant_overfitting():
    pos_enc = np.zeros((8, 4), dtype=np.float32)

    dataset = HOVEncoderDecoderDataset(
        HOVDatasetConfig(
            dir=Path("dummy"),
            seq_len=4,
            sample_stride=1,
        ),
        data=np.array(
            [
                [[1.0, 0.1, 0.9], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, 0.1, 0.5]],
                [[1.0, -0.1, 0.9], [0.0, -0.1, 0.9]],
                [[0.0, 0.0, 0.0], [1.0, 0.1, 0.5]],
                [[1.0, -0.1, 0.9], [0.0, -0.1, 0.9]],
                [[0.0, 0.0, 0.0], [1.0, 0.1, 0.5]],
                [[1.0, 0.1, 0.9], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, 0.1, 0.5]],
            ],
            dtype=np.float32,
        ),
        pos_enc=pos_enc,
    )
    train_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    # Create model and a basic optimizer
    model = DequantTransformer(DequantTransformerConfig(max_seq_len=4, num_instruments=2, d_model=128, n_heads=4, n_layers=2))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(300):
        for _batch_idx, (encoder_input, decoder_input, decoder_target, pos_enc) in enumerate(train_loader):
            output: torch.Tensor = model(encoder_input, decoder_input, pos_enc)
            loss = criterion(output, decoder_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    # Run evaluation on the training dataset
    # Normally, this is a no-no, but because we are intentionally overfitting,
    # we can use the same dataset for training and evaluation
    with torch.no_grad():
        prediction = model.forward(
            encoder_input[0].unsqueeze(0),
            decoder_input[0].unsqueeze(0),
            pos_enc[0].unsqueeze(0),
        )[0].detach()
        eval_loss = criterion(prediction, decoder_target[0])
        print(f"Evaluation loss: {eval_loss.item():.6f}")

    # Make sure that the evaluation loss goes down
    assert eval_loss < 0.07
