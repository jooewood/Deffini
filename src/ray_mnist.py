#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import torch
import pytorch_lightning as pl
from filelock import FileLock
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
import os

class LightningMNISTClassifier(pl.LightningModule):
    """
    This has been adapted from
    https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09
    """

    def __init__(self, config, data_dir=None):
        super(LightningMNISTClassifier, self).__init__()

        self.data_dir = data_dir or os.getcwd()

        self.layer_1_size = config["layer_1_size"]
        self.layer_2_size = config["layer_2_size"]
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = torch.nn.Linear(28 * 28, self.layer_1_size)
        self.layer_2 = torch.nn.Linear(self.layer_1_size, self.layer_2_size)
        self.layer_3 = torch.nn.Linear(self.layer_2_size, 10)

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)

        x = self.layer_1(x)
        x = torch.relu(x)

        x = self.layer_2(x)
        x = torch.relu(x)

        x = self.layer_3(x)
        x = torch.log_softmax(x, dim=1)

        return x

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def accuracy(self, logits, labels):
        _, predicted = torch.max(logits.data, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / len(labels)
        return torch.tensor(accuracy)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)

        self.log("ptl/train_loss", loss)
        self.log("ptl/train_accuracy", accuracy)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)
        return {"val_loss": loss, "val_accuracy": accuracy}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_accuracy"] for x in outputs]).mean()
        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/val_accuracy", avg_acc)

    @staticmethod
    def download_data(data_dir):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ])
        with FileLock(os.path.expanduser("~/.data.lock")):
            return MNIST(data_dir, train=True, download=True, transform=transform)

    def prepare_data(self):
        mnist_train = self.download_data(self.data_dir)

        self.mnist_train, self.mnist_val = random_split(
            mnist_train, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=int(self.batch_size))

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=int(self.batch_size))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def train_mnist(config):
    model = LightningMNISTClassifier(config)
    trainer = pl.Trainer(max_epochs=10, show_progress_bar=False)
    trainer.fit(model)

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.cloud_io import load as pl_load
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback
from ray.tune.schedulers import PopulationBasedTraining

from ray.tune.integration.pytorch_lightning import TuneReportCallback
callback = TuneReportCallback({
    "loss": "avg_val_loss",
    "mean_accuracy": "avg_val_accuracy"
}, on="validation_end")

def train_mnist_tune(config, num_epochs=10, num_gpus=1):
    data_dir = os.path.expanduser("~/data")
    model = LightningMNISTClassifier(config, data_dir)
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        # If fractional GPUs passed in, convert to int.
        gpus=math.ceil(num_gpus),
        logger=TensorBoardLogger(
            save_dir=tune.get_trial_dir(), name="", version="."),
        progress_bar_refresh_rate=0,
        callbacks=[
            TuneReportCallback(
                {
                    "loss": "ptl/val_loss",
                    "mean_accuracy": "ptl/val_accuracy"
                },
                on="validation_end")
        ])
    trainer.fit(model)

def tune_mnist_asha(num_samples=10, num_epochs=10, gpus_per_trial=0.1):
    config = {
        "layer_1_size": tune.choice([32, 64, 128]),
        "layer_2_size": tune.choice([64, 128, 256]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128]),
    }

    # scheduler = ASHAScheduler(
    #     max_t=num_epochs,
    #     grace_period=1,
    #     reduction_factor=2)

    scheduler = PopulationBasedTraining()

    reporter = CLIReporter(
        parameter_columns=["layer_1_size", "layer_2_size", "lr", "batch_size"],
        metric_columns=["loss", "mean_accuracy", "training_iteration"])

    analysis = tune.run(
        tune.with_parameters(
            train_mnist_tune,
            num_epochs=num_epochs,
            num_gpus=gpus_per_trial),
        resources_per_trial={
            "cpu": 1,
            "gpu": gpus_per_trial
        },
        metric="loss",
        mode="min",
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_mnist_PBT")

    print("Best hyperparameters found were: ", analysis.best_config)


if __name__ == "__main__":
    import ray
    ray.init(address='172.22.99.11:6000', _redis_password='5241590000000000')
    tune_mnist_asha()