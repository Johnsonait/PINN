from abc import abstractmethod

from torch import Tensor
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
    def __init__(self) -> None:
        return

    @abstractmethod
    def loss(self,ins,outs,targets) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def __configure_inputs(self,input_batch) -> tuple[Tensor]:
        """This method recieves the input data as a batch and splits it into 
        seperate tensors for each physical input. 
        For example, an input batch may look like [batch_size, features]. Here,
        the "features" are known to the user as physical quantities such as 
        scalars time, pressure,temperature or multidimensional quantities like
        stress or strain. This method should split the features into seperate
        batches where each batch is a physical quantity. If the input features
        are a set of physical quanties like features := [time,strain]
        then the full feature vector is [batch_size, 1+6]. This function should
        return two tensors: 
        time = [batch_size,1] = input_batch[:,0]
        strain = [batch_size,6] = input_batch[:,1:]
        Note that the class should also try to explicitly keep track of which 
        index refers to which physical quantity to avoid hard-coding index values
        like was done above."""
        raise NotImplementedError

    @abstractmethod
    def __configure_outputs(self,output_batch) -> tuple[Tensor]:
        raise NotImplementedError

    @abstractmethod
    def configure_dataset(self,dataset: Dataset) -> tuple[DataLoader,DataLoader]:
        """Method that configures the incoming dataset into a pair of 
        training and validation dataloaders"""
        raise NotImplementedError

