import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from typing import Optional
from datetime import datetime
from ..utils.checkpoint import Checkpoint
from ..data.datasets.hov_dataset import HOVEncoderDecoderDataset, HOVDatasetConfig
from ..model import (
    DequantTransformer as DequantTransformer,
    DequantTransformerConfig as DequantTransformerConfig,
)
from ..config import CONFIG

logger = logging.getLogger("training")


class Trainer:
    def __init__(self):
        # Set the device
        if CONFIG.train.device is None:
            # Choose device automatically based on availability
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            # Use the configured device
            self.device = torch.device(CONFIG.train.device)

        logger.info(f"Using device '{self.device}'")

        self.train_set = self.create_dataloader(CONFIG.data.dir / "train")
        self.test_set = self.create_dataloader(CONFIG.data.dir / "test")
        self.validation_set = self.create_dataloader(CONFIG.data.dir / "validation")

        self.model = DequantTransformer(
            DequantTransformerConfig(
                max_seq_len=CONFIG.model.max_seq_len,
                num_instruments=CONFIG.model.drums.num_instruments,
                d_model=CONFIG.model.transformer.d_model,
            )
        )
        self.model = self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=CONFIG.train.learning_rate)
        self.loss_fn = torch.nn.MSELoss()

        self.epoch = 0

    def train_epoch(self):
        # Training
        logger.info("Training ...")
        self.model.train()

        for encoder_input, decoder_input, decoder_target in self.train_set:
            encoder_input = encoder_input.to(self.device)
            decoder_input = decoder_input.to(self.device)
            decoder_target = decoder_target.to(self.device)

            # Forward pass
            predictions = self.model(encoder_input, decoder_input)

            # Compute loss
            loss: torch.Tensor = self.loss_fn(predictions, decoder_target)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Valiation
        logger.info("Validating ...")
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for encoder_input, decoder_input, decoder_target in self.validation_set:
                encoder_input = encoder_input.to(self.device)
                decoder_input = decoder_input.to(self.device)
                decoder_target = decoder_target.to(self.device)

                predictions = self.model(encoder_input, decoder_input)

                loss = self.loss_fn(predictions, decoder_target)
                total_loss += loss.item()
                num_batches += 1

        # Update epoch
        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {self.epoch + 1}/{CONFIG.train.num_epochs} - Loss: {avg_loss:.6f}")
        self.epoch += 1

        # Save a checkpoint to the default path
        if (self.epoch + 1) % CONFIG.train.save_every_n_epochs == 0:
            self.save_checkpoint()

    def train(self):
        # Resume from a previous checkpoint
        if CONFIG.train.resume:
            try:
                self.load_checkpoint(CONFIG.train.resume_from)
            except FileNotFoundError as e:
                if CONFIG.train.resume_from is None:
                    logger.warning("Could not resume, because no checkpoint could be loaded.")
                else:
                    raise e

            logger.info(f"Resuming training from epoch {self.epoch} ...")

        # Train until the specified epoch
        while self.epoch < CONFIG.train.num_epochs:
            self.train_epoch()

    @staticmethod
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

    def save_checkpoint(self, filename: Optional[Path] = None):
        # Generate a filename if none was provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = CONFIG.train.checkpoint_dir / f"cp_{timestamp}.pt"

        # Save the checkpoint
        Checkpoint.save(
            filename,
            config=CONFIG,
            epoch=self.epoch,
            model=self.model,
            optimizer=self.optimizer,
            loss_fn=self.loss_fn,
        )

    def load_checkpoint(self, filename: Optional[Path] = None):
        # Use latest checkpoint if none was provided
        if not filename:
            # Search for the latest checkpoint file
            filename = max(CONFIG.train.checkpoint_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, default=None)

            if filename is None:
                raise FileNotFoundError("No checkpoints were found.")

        # Load the checkpoint and apply all parameters
        self.epoch, self.loss_fn = Checkpoint.load(
            filename,
            device=self.device,
            config=CONFIG,
            model=self.model,
            optimizer=self.optimizer,
        )
