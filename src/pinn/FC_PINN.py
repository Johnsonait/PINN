import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset,DataLoader

from pinn import PINN


class SimplePINN(PINN):
    """Example implementation of a PINN that uses simple fully connected layers
    as its main architecture"""

    __INS = {
        'time' : 0,
        'strain' : slice(1,7),
        'velocity_grad' : slice(7,16)
    }
    __num_in = 1+6+9

    __OUTS = {
        'stress' : slice(0,6),
        'tau'    : slice(6,18),
        'gam'    : slice(18,30),
        'De'     : slice(30,36)
    }

    __num_out = 6+12+12+6

    def __init__(self,
                depth = 1,
                width = 1,
                device = torch.device('cpu')                
                ) -> None:
        super().__init__(self,SimplePINN)

        self.__depth  = depth
        self.__width  = width
        self.__device = device

        self.net = nn.Sequential()
        self.net.append(nn.Linear(self.__num_in,width))
        self.net.append(nn.ReLU)
        for layer in range(depth):
            self.net.append(nn.Linear(width,width))
            self.net.append(nn.ReLU)
        self.net.append(nn.Linear(width,self.__num_out))
        self.net.append(nn.ReLU)

    def loss(self, ins, outs, targets,loss_fcn) -> Tensor:
        
        time,strain,vel_grad = self.__configure_inputs(ins)

        stress,tau,gam,De    = self.__configure_outputs(outs)


        return super().loss(ins, outs, targets)

    def forward(self,x) -> tuple(Tensor):

        assert x.shape[0]<=2,"Badly dimensioned input!"

        outs = self.net(x)

        return outs

    def __configure_inputs(self, input_batch) -> tuple[Tensor]:
        
        time     = input_batch[:,self.__INS['time']]
        strain   = input_batch[:,self.__INS['strain']]
        vel_grad = input_batch[:,self.__INS['velocity_grad']]

        return time,strain,vel_grad

    def __configure_outputs(self, output_batch) -> tuple[Tensor]:
        
        stress = output_batch[:,self.__OUTS['stress']]
        tau = output_batch[:,self.__OUTS['tau']]
        gam = output_batch[:,self.__OUTS['gam']]
        De = output_batch[:,self.__OUTS['De']]

        return stress,tau,gam,De

    def configure_dataset(self, dataset: Dataset) -> tuple[DataLoader]:
        train_dataloader = DataLoader(Dataset)
        val_dataloader = DataLoader(Dataset)

        return train_dataloader, val_dataloader