from abc import abstractmethod

import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn


class PINN(nn.Module):
    """The main base class for defining various forms of "Physics Informed 
    Neural Networks (PINN). 
    The central purpose here is to maintain many of the same interface features
    of the nn.Module class but with some extra functionality that allows for 
    easier prototyping and variation tailored to the needs of PINNs.
    The added functionalities currently are (in the form of class methods)

    - configure_dataset: PINN modules can be (for now) either fully-connected
        or recurrent architectures. These two architectures need some specific
        approaches to handling and preparing their data prior to training. """
    def __init__() -> None:
        pass

    @abstractmethod
    def configure_dataset(self,dataset: Dataset) -> tuple(DataLoader,DataLoader):
        """Method that configures the incoming dataset into a pair of 
        training and validation dataloaders"""
        pass
