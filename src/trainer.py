import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer, SGD
from torch.utils.data import Dataset, DataLoader

from early_stopper import EarlyStopper
from pinn.pinn import PINN

class Trainer():
    
    def __init__(
                    self,
                    options: dict[str,float] = {'lr':1e-3},
                    epochs: int = 1000,
                    model: PINN = PINN,
                    loss_fcn: nn.Module = nn.MSELoss,
                    optimzer: Optimizer = SGD,
                    stopper: EarlyStopper = EarlyStopper,
                    dataset: Dataset = Dataset
                ) -> None:

        self._options : dict[str,float] = options
        self._epochs: int = epochs
        self._model: PINN = model
        self._loss_fcn: nn.Module = loss_fcn
        self._optimizer: Optimizer = optimzer(model.parameters(),**options)
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
        # The PINN model should have configured the dataloaders into batches that
        # conform  to their expected forward() method.
        training_loader,validation_loader: tuple(DataLoader,DataLoader) = self._model.configure_dataset(self._dataset)

        train_loss = 0
        count = 0
        self._model.train()
        for input_batch,target_batch in training_loader:
            count += 1
            pred = self._model(input_batch)

            train_loss += self._loss_fcn(pred,target_batch)

            self._optimizer.step()

            self._optimizer.zero_grad()
        train_loss /= count

        val_loss = 0
        count = 0
        self._model.eval()
        with torch.no_grad():
            for input_batch,target_batch in validation_loader:
                count += 1
                pred = self._model(input_batch)

                val_loss += self._loss_fcn(pred,target_batch)
        val_loss /= count

        return train_loss,val_loss