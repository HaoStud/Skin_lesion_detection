import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

device = torch.device('cuda:0')

class Experiment:
    def __init__(self, model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=10, comment=""):
        self.model       = model
        self.criterion   = criterion
        self.optimizer   = optimizer
        self.scheduler   = scheduler
        self.num_epochs  = num_epochs
        self.comment     = comment
        self.dataloaders = dataloaders
        self.dataset_sizes = dataset_sizes

        self._current_epoch = 0
        self._path = f"output/{comment}"
        self._tb = SummaryWriter(log_dir=self._path, comment=self.comment)

        self.best_acc = 0.0
        self.best_loss = 0.0

        if not os.path.exists(self._path): os.makedirs(self._path)


    def train_model(self):
        print(f'Training {self.comment} ...')
        for _ in tqdm(range(self.num_epochs)):
            self.__train__()
            self.__validate__()
            self._current_epoch += 1
        self.__save_model__()
        self._tb.close()
        print(f'Best validation accuracy: {self.best_acc:4f}')


    def __train__(self):
        self.phase = 'train'
        self.model.train()

        running_loss     = 0.0
        running_corrects = 0

        for inputs, labels in self.dataloaders['train']:
            inputs = inputs.to(device);
            labels = labels.to(device)

            self.optimizer.zero_grad()

            # forward
            outputs  = self.model(inputs)
            _, preds = torch.max(outputs, 1)
            loss     = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            # statistics
            running_loss     += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        self.epoch_loss = running_loss / self.dataset_sizes['train']
        self.epoch_acc  = running_corrects.double() / self.dataset_sizes['train']

        self.scheduler.step()
        self.__tensorboard__()


    def __validate__(self):
        self.phase = 'val'
        self.model.eval()

        running_loss     = 0.0
        running_corrects = 0

        for inputs, labels in self.dataloaders['val']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            self.optimizer.zero_grad()

            # forward
            outputs  = self.model(inputs)
            _, preds = torch.max(outputs, 1)
            loss     = self.criterion(outputs, labels)

            # statistics
            running_loss     += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        self.epoch_loss = running_loss / self.dataset_sizes['val']
        self.epoch_acc  = running_corrects.double() / self.dataset_sizes['val']

        # deep copy the model
        if self.epoch_acc > self.best_acc:
            self.best_acc       = self.epoch_acc
            self.best_loss      = self.epoch_loss
            self.__save_model__(is_best=True)
        self.__tensorboard__()


    def __tensorboard__(self):
        self._tb.add_scalar(f"Loss/{self.phase}",     self.epoch_loss, self._current_epoch)
        self._tb.add_scalar(f"Accuracy/{self.phase}", self.epoch_acc,  self._current_epoch)


    def __save_model__(self, is_best=False):
        model_name = "best_model.pt" if is_best else "model.pt"
        path       = f"{self._path}/{model_name}"

        torch.save({
            'epoch': self._current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.epoch_loss
        }, path)
