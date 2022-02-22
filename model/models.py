#!/bin/env pipenv-shebang

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
import torchvision.models as models
import pytorch_lightning as pl
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_lightning import loggers as pl_loggers

from model.monte_carlo import add_dropout_layers
from data.dataloader import HAM10000


# Name to model dict
model_names = {
    'resnet': 'resnet18',
    'resnet-drop': 'resnet50-drop',
    'alexnet': 'alexnet',
    'vgg': 'vgg11_bn',
    'squezzenet': 'squeezenet1_0',
    'densenet': 'densenet121',
}


class ImgClassificationModel(pl.LightningModule):
    def __init__(self, num_classes=7, lr=1e-3, batch_size=8, name='resnet'):
        super().__init__()
        self.lr = lr
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy()
        self.save_hyperparameters()

        # Load model respective model and rewrite last layer
        if name == 'resnet':
            self.model = models.resnet18(pretrained=True)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        if name == 'resnet-drop':
            self.model = models.resnet50(pretrained=True)
            self.model.fc = nn.Sequential(
                nn.Dropout(p=0.5, inplace=True),
                nn.Linear(self.model.fc.in_features, num_classes))
            add_dropout_layers(self.model)
        if name == 'alexnet':
            self.model = models.alexnet(pretrained=True)
            self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, num_classes)
        if name == 'vgg':
            self.model = models.vgg11_bn(pretrained=True)
            self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, num_classes)
        if name == 'squeezenet':
            self.model = models.squeezenet1_0(pretrained=True)
            self.model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        if name == 'densenet':
            self.model = models.densenet121(pretrained=True)
            self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_epoch_end(self, outs):
        print('')

    def training_step(self, batch, batch_idx):
        imgs, label = batch

        out = self(imgs)
        loss = self.criterion(out, label)

        self.log('train_loss', loss, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, label = batch

        out = self(imgs)
        loss = self.criterion(out, label)

        pred = torch.argmax(out, dim=1)
        self.accuracy(pred, label)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def setup(self, stage=None):

        self.train_ds = HAM10000('input/aug/train')
        self.val_ds   = HAM10000('input/aug/val')
        self.test_ds  = HAM10000('input/aug/test')

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=2)


# Create model, pick one of the following names: resnet, alexnet, vgg, squeezenet, densenet:
def train(model_name='resnet', bs=16, num_epochs=20):
    BATCH_SIZE = bs
    LEARNING_RATE = 1e-4
    model = ImgClassificationModel(name=model_name, batch_size=BATCH_SIZE, lr=LEARNING_RATE)

    # Validation loss checkpoint
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath=f'output/checkpoints',
        filename=f'{model_name}_{{epoch:02d}}_{{val_acc:.2f}}',
        save_top_k=1,
        mode='max',
    )

    # Custom logger
    tb_logger = pl_loggers.TensorBoardLogger("output/logs")

    # Initialize trainer and start training
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=num_epochs,
        callbacks=[checkpoint_callback],
        logger=tb_logger,
        accumulate_grad_batches=1,
    )

    trainer.fit(model)
    # trainer.test(model)
