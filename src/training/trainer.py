import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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

        self.train_set = self.create_dataloader(CONFIG.data.dir / "train", CONFIG.train.max_train_samples)
        self.test_set = self.create_dataloader(CONFIG.data.dir / "test", CONFIG.train.max_test_samples)
        self.validation_set = self.create_dataloader(CONFIG.data.dir / "validation", CONFIG.train.max_val_samples)

        self.model = DequantTransformer(
            DequantTransformerConfig(
                max_seq_len=CONFIG.model.max_seq_len,
                num_instruments=CONFIG.model.drums.num_instruments,
                d_model=CONFIG.model.transformer.d_model,
                n_heads=CONFIG.model.transformer.n_heads,
                n_layers=CONFIG.model.transformer.n_layers,
                dropout=CONFIG.model.transformer.dropout,
            )
        )
        self.model = self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=CONFIG.train.learning_rate,
            weight_decay=CONFIG.train.weight_decay,
        )
        self.loss_fn = torch.nn.MSELoss()

        self.scheduler = self._create_scheduler()

        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.writer = SummaryWriter()

    def train_epoch(self):
        # Training
        logger.info("Training ...")
        self.model.train()

        # Show a progress bar
        pbar = tqdm(self.train_set, desc=f"Epoch {self.epoch}", mininterval=0.1)

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
        self.writer.add_scalar("Loss/validation", avg_loss, self.global_step)

        # Best model tracking
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.patience_counter = 0
            self.save_checkpoint(CONFIG.train.checkpoint_dir / "best.pt")
            logger.info(f"New best validation loss: {avg_loss:.6f}")
        else:
            self.patience_counter += 1

        # Save a periodic checkpoint
        if self.epoch % CONFIG.train.save_every_n_epochs == 0:
            self.save_checkpoint()

        logger.info(f"Epoch {self.epoch + 1}/{CONFIG.train.num_epochs} - Val loss: {avg_loss:.6f} (best: {self.best_val_loss:.6f}, patience: {self.patience_counter})")

        # Step the learning rate scheduler
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(avg_loss)
            else:
                self.scheduler.step()

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

            # Early stopping
            patience = CONFIG.train.early_stopping_patience
            if patience > 0 and self.patience_counter >= patience:
                logger.info(f"Early stopping: no improvement for {patience} epochs.")
                break

        # Close the tensorboard writer
        self.writer.close()

    def _create_scheduler(self):
        name = CONFIG.train.lr_scheduler
        if name == "none":
            return None
        elif name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=CONFIG.train.num_epochs - CONFIG.train.lr_warmup_epochs
            )
        elif name == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=3
            )
        else:
            raise ValueError(f"Unknown lr_scheduler: {name}")

        if CONFIG.train.lr_warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=0.01, total_iters=CONFIG.train.lr_warmup_epochs
            )
            if name == "plateau":
                return warmup  # SequentialLR doesn't support ReduceLROnPlateau; just use warmup then step plateau manually
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer, schedulers=[warmup, scheduler], milestones=[CONFIG.train.lr_warmup_epochs]
            )
        return scheduler

    @staticmethod
    def create_dataloader(dir: Path, max_samples: Optional[int] = None):
        logger.info(f"Loading dataset '{dir}' ...")

        dataset = HOVEncoderDecoderDataset(
            HOVDatasetConfig(
                dir=dir,
                seq_len=CONFIG.model.max_seq_len,
                sample_stride=CONFIG.train.sample_stride,
                filter_empty=False,
                max_samples=max_samples,
            )
        )
        dataloader = DataLoader(
            dataset,
            batch_size=CONFIG.train.batch_size,
            shuffle=CONFIG.train.sample_shuffle,
            pin_memory=True if torch.cuda.is_available() else False,
            num_workers=0,
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
            best_val_loss=self.best_val_loss,
        )

    def load_checkpoint(self, filename: Optional[Path] = None):
        # Use latest checkpoint if none was provided
        if not filename:
            # Search for the latest checkpoint file
            filename = max(CONFIG.train.checkpoint_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, default=None)

            if filename is None:
                raise FileNotFoundError("No checkpoints were found.")

        # Load the checkpoint and apply all parameters
        self.epoch, self.loss_fn, self.global_step, self.best_val_loss = Checkpoint.load(
            filename,
            device=self.device,
            config=CONFIG,
            model=self.model,
            optimizer=self.optimizer,
        )
