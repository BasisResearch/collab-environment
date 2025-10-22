"""
Trainer for GraphGym models on boids trajectory prediction.

Implements training loop, evaluation, and metrics for node-level regression.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from loguru import logger
from tqdm import tqdm
import json

from .models import GraphGymGNN
from .config import ExperimentConfig
from .dataset import BoidsGraphGymDataset


class GraphGymTrainer:
    """
    Trainer for GraphGym models on trajectory prediction.

    Metrics:
    - MSE (Mean Squared Error): Primary metric
    - MAE (Mean Absolute Error): Alternative metric
    - R² (Coefficient of Determination): Goodness of fit
    """

    def __init__(
        self,
        model: GraphGymGNN,
        config: ExperimentConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        self.model = self.model.to(self.device)
        logger.info(f"Using device: {self.device}")

        # Loss function
        self.criterion = nn.MSELoss()

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

        # Metrics history
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []

        # Output directory
        self.out_dir = Path(config.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer from config."""
        opt_config = self.config.optim

        if opt_config.optimizer.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=opt_config.base_lr,
                weight_decay=opt_config.weight_decay,
            )
        elif opt_config.optimizer.lower() == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=opt_config.base_lr,
                momentum=0.9,
                weight_decay=opt_config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config.optimizer}")

    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler from config."""
        opt_config = self.config.optim

        if opt_config.scheduler == "reduce_on_plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=opt_config.reduce_factor,
                patience=opt_config.schedule_patience,
                min_lr=opt_config.min_lr,
            )
        elif opt_config.scheduler == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.5,
            )
        elif opt_config.scheduler == "none":
            return None
        else:
            logger.warning(f"Unknown scheduler: {opt_config.scheduler}, using none")
            return None

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch in pbar:
            batch = batch.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            out = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            # Compute loss
            loss = self.criterion(out, batch.y)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track loss
            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on a dataset.

        Returns:
            Dictionary of metrics
        """
        self.model.eval()

        total_loss = 0
        total_mae = 0
        all_preds = []
        all_targets = []
        num_batches = 0

        for batch in loader:
            batch = batch.to(self.device)

            # Forward pass
            out = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            # Compute loss
            loss = self.criterion(out, batch.y)
            mae = torch.abs(out - batch.y).mean()

            total_loss += loss.item()
            total_mae += mae.item()
            num_batches += 1

            # Collect predictions for R²
            all_preds.append(out.cpu())
            all_targets.append(batch.y.cpu())

        # Average metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_mae = total_mae / num_batches if num_batches > 0 else 0

        # Compute R²
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        r2 = self._compute_r2(all_preds, all_targets)

        metrics = {
            'mse': avg_loss,
            'mae': avg_mae,
            'r2': r2,
        }

        return metrics

    def _compute_r2(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Compute R² (coefficient of determination).

        R² = 1 - (SS_res / SS_tot)
        where SS_res = sum of squared residuals
              SS_tot = total sum of squares
        """
        ss_res = torch.sum((targets - predictions) ** 2)
        ss_tot = torch.sum((targets - targets.mean()) ** 2)

        if ss_tot == 0:
            return 0.0

        r2 = 1 - (ss_res / ss_tot)
        return r2.item()

    def train(self) -> Dict[str, List[float]]:
        """
        Full training loop.

        Returns:
            Dictionary of training history
        """
        logger.info("Starting training...")
        logger.info(f"Total epochs: {self.config.optim.max_epoch}")

        for epoch in range(self.config.optim.max_epoch):
            self.current_epoch = epoch

            # Train for one epoch
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.6f}")

            # Validation
            if self.val_loader is not None and epoch % self.config.train.eval_period == 0:
                val_metrics = self.evaluate(self.val_loader)
                self.val_losses.append(val_metrics['mse'])
                self.val_metrics.append(val_metrics)

                logger.info(f"Epoch {epoch}: Val MSE = {val_metrics['mse']:.6f}, "
                           f"MAE = {val_metrics['mae']:.6f}, R² = {val_metrics['r2']:.4f}")

                # Learning rate scheduler
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['mse'])
                    else:
                        self.scheduler.step()

                # Early stopping
                if self.config.train.early_stopping:
                    if val_metrics['mse'] < self.best_val_loss - self.config.train.early_stopping_min_delta:
                        self.best_val_loss = val_metrics['mse']
                        self.epochs_without_improvement = 0
                        # Save best model
                        self.save_checkpoint(is_best=True)
                    else:
                        self.epochs_without_improvement += 1

                    if self.epochs_without_improvement >= self.config.train.early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break

            # Checkpoint saving
            if self.config.train.enable_ckpt and epoch % self.config.train.ckpt_period == 0:
                self.save_checkpoint()

        logger.info("Training complete!")

        # Final test evaluation
        if self.test_loader is not None:
            test_metrics = self.evaluate(self.test_loader)
            logger.info(f"Test Metrics: MSE = {test_metrics['mse']:.6f}, "
                       f"MAE = {test_metrics['mae']:.6f}, R² = {test_metrics['r2']:.4f}")

        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics,
        }

        return history

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }

        if is_best:
            path = self.out_dir / 'best_model.pt'
            logger.info(f"Saving best model to {path}")
        else:
            path = self.out_dir / f'checkpoint_epoch_{self.current_epoch}.pt'

        torch.save(checkpoint, path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.val_metrics = checkpoint.get('val_metrics', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")

    def save_metrics(self, filename: str = "metrics.json"):
        """Save metrics to JSON file."""
        metrics = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics,
            'best_val_loss': self.best_val_loss,
            'final_epoch': self.current_epoch,
        }

        path = self.out_dir / filename
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Saved metrics to {path}")


def train_model(
    config: ExperimentConfig,
    train_dataset: BoidsGraphGymDataset,
    val_dataset: Optional[BoidsGraphGymDataset] = None,
    test_dataset: Optional[BoidsGraphGymDataset] = None,
) -> Tuple[GraphGymGNN, Dict[str, List[float]]]:
    """
    Train a GraphGym model.

    Args:
        config: Experiment configuration
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        test_dataset: Test dataset (optional)

    Returns:
        (trained_model, training_history)
    """
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.train.batch_size,
            shuffle=False,
            num_workers=0,
        )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.train.batch_size,
            shuffle=False,
            num_workers=0,
        )

    # Create model
    from .models import create_model_from_config

    # Update in_dim based on actual dataset
    sample = train_dataset[0]
    config.gnn.in_dim = sample.x.shape[1]

    model = create_model_from_config(config)

    # Create trainer
    trainer = GraphGymTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    # Train
    history = trainer.train()

    # Save metrics
    trainer.save_metrics()

    return model, history
