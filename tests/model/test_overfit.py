import torch
import torch.nn as nn
from src.model import Dequant, DequantConfig


def test_dequant_overfitting():
    # Dataset shape: (batch, seq_len, instrument, hov=3)
    dataset_input = torch.tensor(
        [
            [
                [[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]],
                [[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
                [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]],
            ],
            [
                [[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
                [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]],
                [[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]],
            ],
        ]
    )
    dataset_target = torch.tensor(
        [
            [
                [[1.0, 0.1, 0.9], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, 0.1, 0.5]],
                [[1.0, -0.1, 0.9], [1.0, -0.1, 0.9]],
                [[0.0, 0.0, 0.0], [1.0, 0.1, 0.5]],
            ],
            [
                [[1.0, -0.1, 0.9], [1.0, -0.1, 0.9]],
                [[0.0, 0.0, 0.0], [1.0, 0.1, 0.5]],
                [[1.0, 0.1, 0.9], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, 0.1, 0.5]],
            ],
        ]
    )

    # Dissect the overfitting dataset into model input and target
    encoder_input = dataset_input
    decoder_input = torch.cat(
        [
            torch.zeros(2, 1, 2, 3),  # Start token
            dataset_target[:, :-1],  # Shift right along the seq_len dimension
        ],
        dim=1,
    )
    decoder_target = dataset_target

    # Create model and a basic optimizer
    model = Dequant(DequantConfig(max_seq_len=4, num_instruments=2))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(1000):
        output: torch.Tensor = model(encoder_input, decoder_input)
        loss = criterion(output, decoder_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    # Run evaluation on the first dataset sample
    # Normally, this is a no-no, but because we are intentionally overfitting,
    # we can use the same dataset for training and evaluation
    with torch.no_grad():
        prediction = model.forward(
            encoder_input[0].unsqueeze(0),
            decoder_input[0].unsqueeze(0),
        )[0].detach()
        eval_loss = criterion(prediction, decoder_target[0])
        print(f"Evaluation loss: {eval_loss.item():.6f}")

    # Make sure that the evaluation loss goes down
    assert eval_loss < 0.05
