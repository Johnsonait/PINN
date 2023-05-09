from numpy import inf

class EarlyStopper():
    
    def __init__(self,patience: int =20,delta: float =1e-3) -> None:
        self._patience: int = patience
        self._delta: float = delta
        self._count: int = 0
        self._min_loss: float = inf
    
    def must_stop(self,loss: float) -> bool:
        if loss > self._min_loss:
            self._count += 1
        elif loss < self._min_loss:
            self._min_loss = loss
            self._count = 0
        
        if self._count < self._patience:
            return True
        else:
             return False 
