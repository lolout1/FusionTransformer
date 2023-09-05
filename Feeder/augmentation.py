from typing import Any, Tuple, Dict
import random
import numpy as np 
import torch 


class TimeDrift(object):
    def __init__(self, drift) -> None:
        self.drift = drift
        random_walk = list()
        random_walk.append(-drift if random() < 0.5 else drift)
    
    def __call__(self,data:Dict[str,torch.tensor], label:torch.tensor) -> Tuple[Dict[str, torch.tensor], torch.tensor]:
            acc_rand = np.random.rand(data['acc_data'].shape[0])
            acc_rand = acc_rand[acc_rand>0] * -1
            acc_rand[acc_rand==0]  = 1

class TSFilpper(object):
    def __init__(self):
        print('Initiating the Flipper class')
    
    def __call__(self, acc_data, skl_data):
        acc_data = -1 * acc_data
        skl_data = -1 * skl_data

        return acc_data, skl_data