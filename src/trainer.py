import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer, SGD
from torch.utils.data import Dataset

from early_stopper import EarlyStopper

class Trainer():
    
    def __init__(
                    self,
                    lr: float = 1e-3,
                    epochs: int = 1000,
                    model: nn.Module = nn.Module,
                    optimzer: Optimizer = SGD,
                    stopper: EarlyStopper = EarlyStopper,
                    dataset: Dataset = Dataset
                ) -> None:

        self._lr: float = lr
        self._epochs: int = epochs
        self._model: nn.Module = model
        self._optimizer: Optimizer = optimzer
        self._stopper: EarlyStopper = stopper
        self._dataset: Dataset = dataset

        return

    def fit(self) -> None:
        for epoch in range(self._epochs):
            print(f'Epoch: {epoch}')

            train_loss, val_loss = self.fit_epoch()
            print(f'Training loss: {train_loss}')
            print(f'Validation loss: {val_loss}')

            if self._stopper.must_stop(val_loss):
                print(f'Reached early stopping criteria at epoch: {epoch}')
                return
            

        return

    def fit_epoch(self) -> tuple(float,float):
        
        return