import torch
import torch.nn as nn

import torch.nn as nn


class Generator(nn.Module):

    def __init__(self,seedDim):
        super(Generator,self).__init__()

        self.M1 = nn.Sequential(
            nn.Linear(seedDim,64*64),
            nn.LeakyReLU(0.2),
            nn.Linear(64*64,32*32,),
            nn.LeakyReLU(0.2),
            nn.Linear(32*32,16*16),
            nn.LeakyReLU(0.2),
            nn.Linear(16*16,28*28),
            nn.Tanh(),
        )
        for m in self.M1:
            if isinstance(m, (nn.Linear)):
                nn.init.normal_(m.weight, mean=0, std=0.02)
    
    def forward(self,X):
        X = self.M1(X)
        return X

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator,self).__init__()

        self.M1 = nn.Sequential(
            nn.Linear(28*28 ,20*20),
            nn.LeakyReLU(0.2),
            
            nn.Linear(20*20, 12*12),
            nn.LeakyReLU(0.2),
        

            nn.Linear(12*12, 4*4),
            nn.LeakyReLU(0.2),

            nn.Linear(4*4, 1),
            nn.Sigmoid(),
            )
        for m in self.M1:
            if isinstance(m, (nn.Linear)):
                nn.init.normal_(m.weight, mean=0, std=0.02)
        
    def forward(self,X):
        X = self.M1(X)
        return X

