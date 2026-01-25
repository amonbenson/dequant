import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from ..hov.dataset import HOVEncoderDecoderDataset, HOVDatasetConfig
from ..model import (
    DequantTransformer as DequantTransformer,
    DequantTransformerConfig as DequantTransformerConfig,
)
from ..config import CONFIG

logger = logging.getLogger("train")


def save_checkpoint(path, model, optimizer, scheduler, epoch, loss_fn):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss_fn": loss_fn,
        },
        path,
    )


def load_checkpoint(path, model, optimizer, scheduler, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    epoch = checkpoint["epoch"]
    loss_fn = checkpoint["loss_fn"]
    return epoch, loss_fn


def create_dataloader(dir: Path):
    logger.info(f"Loading dataset '{dir}' ...")

    dataset = HOVEncoderDecoderDataset(
        HOVDatasetConfig(
            dir=dir,
            seq_len=CONFIG.model.max_seq_len,
            sample_stride=CONFIG.train.sample_stride,
            filter_empty=True,
        )
    )
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG.train.batch_size,
        shuffle=CONFIG.train.sample_shuffle,
    )
    return dataloader


def train():
    device = CONFIG.train.device
    if device is None:
        # Choose device automatically based on availability
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        # Use the configured device
        device = torch.device(device)

    logger.info(f"Using device '{device}'")

    train_set, test_set, validation_set = [create_dataloader(CONFIG.data.dir / split_name) for split_name in ("train", "test", "validation")]

    model = DequantTransformer(
        DequantTransformerConfig(
            max_seq_len=CONFIG.model.max_seq_len,
            num_instruments=CONFIG.model.drums.num_instruments,
            d_model=CONFIG.model.transformer.d_model,
        )
    )
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG.train.learning_rate)
    loss_fn = torch.nn.MSELoss()

    num_epochs = CONFIG.train.num_epochs
    for epoch in range(num_epochs):
        # Training
        logger.info("Running training...")
        model.train()

        for encoder_input, decoder_input, decoder_target in train_set:
            encoder_input = encoder_input.to(device)
            decoder_input = decoder_input.to(device)
            decoder_target = decoder_target.to(device)

            # Forward pass
            predictions = model(encoder_input, decoder_input)

            # Compute loss
            loss = loss_fn(predictions, decoder_target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Valiation
        logger.info("Running validation...")
        model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for encoder_input, decoder_input, decoder_target in validation_set:
                encoder_input = encoder_input.to(device)
                decoder_input = decoder_input.to(device)
                decoder_target = decoder_target.to(device)

                predictions = model(encoder_input, decoder_input)

                loss = loss_fn(predictions, decoder_target)
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.6f}")
