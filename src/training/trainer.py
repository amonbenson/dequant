import logging
import multiprocessing
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from ..config import CONFIG
from ..data.datasets.hov_dataset import HOVDatasetConfig, HOVEncoderDecoderDataset
from ..model import DequantTransformer as DequantTransformer
from ..model import DequantTransformerConfig as DequantTransformerConfig
from ..utils.checkpoint import Checkpoint

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
                dropout=CONFIG.model.transformer.dropout,
            )
        )
        self.model = self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=CONFIG.train.learning_rate)
        self.loss_fn = torch.nn.MSELoss()

        self.epoch = 0
        self.global_step = 0
        self.writer = SummaryWriter()

    def train_epoch(self):
        # Training
        logger.info("Training ...")
        self.model.train()

        # Show a progress bar
        pbar = tqdm(self.train_set, desc=f"Epoch {self.epoch}", mininterval=0.1, file=sys.stdout, disable=False, leave=False)

        # Start training batches
        for encoder_input, decoder_input, decoder_target, pos_enc in pbar:
            encoder_input = encoder_input.to(self.device, non_blocking=True)
            decoder_input = decoder_input.to(self.device, non_blocking=True)
            decoder_target = decoder_target.to(self.device, non_blocking=True)
            pos_enc = pos_enc.to(self.device, non_blocking=True)

            # Forward pass
            predictions = self.model(encoder_input, decoder_input, pos_enc)

            # Compute loss
            loss: torch.Tensor = self.loss_fn(predictions, decoder_target)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Log training metrics
            self.writer.add_scalar("Loss/train", loss.item(), self.global_step)

            # Log gradient norms for monitoring training stability
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5
            self.writer.add_scalar("Gradients/norm", total_norm, self.global_step)

            # Log learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("Learning_rate", current_lr, self.global_step)

            pbar.set_postfix(loss=f"{loss.item():.4f}", grad=f"{total_norm:.2f}", lr=f"{current_lr:.2e}")

            self.global_step += 1

        # Validation
        logger.info("Validating ...")
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for encoder_input, decoder_input, decoder_target, pos_enc in self.validation_set:
                encoder_input = encoder_input.to(self.device, non_blocking=True)
                decoder_input = decoder_input.to(self.device, non_blocking=True)
                decoder_target = decoder_target.to(self.device, non_blocking=True)
                pos_enc = pos_enc.to(self.device, non_blocking=True)

                predictions = self.model(encoder_input, decoder_input, pos_enc)

                loss = self.loss_fn(predictions, decoder_target)
                total_loss += loss.item()
                num_batches += 1

        # Calculate and log average validation loss
        avg_loss = total_loss / num_batches
        self.writer.add_scalar("Loss/validation", avg_loss, self.epoch)

        # Save a checkpoint to the default path
        if self.epoch % CONFIG.train.save_every_n_epochs == 0:
            self.save_checkpoint()

        logger.info(f"Epoch {self.epoch + 1}/{CONFIG.train.num_epochs} - Loss: {avg_loss:.6f}")
        self.epoch += 1

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

        logger.info(f"Starting training from epoch {self.epoch} ...")

        # Train until the specified epoch
        while self.epoch < CONFIG.train.num_epochs:
            self.train_epoch()

        # Close the tensorboard writer
        self.writer.close()

    @staticmethod
    def create_dataloader(dir: Path):
        logger.info(f"Loading dataset '{dir}' ...")

        dataset = HOVEncoderDecoderDataset(
            HOVDatasetConfig(
                dir=dir,
                seq_len=CONFIG.model.max_seq_len,
                sample_stride=CONFIG.train.sample_stride,
                filter_empty=False,
            )
        )
        dataloader = DataLoader(
            dataset,
            batch_size=CONFIG.train.batch_size,
            shuffle=CONFIG.train.sample_shuffle,
            pin_memory=True if torch.cuda.is_available() else False,
            num_workers=multiprocessing.cpu_count(),  # Adjust based on your CPU cores
            persistent_workers=True,  # Keep workers alive between epochs
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
            global_step=self.global_step,
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
        self.epoch, self.loss_fn, self.global_step = Checkpoint.load(
            filename,
            device=self.device,
            config=CONFIG,
            model=self.model,
            optimizer=self.optimizer,
        )
