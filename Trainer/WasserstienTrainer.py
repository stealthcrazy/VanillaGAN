import torch
import torch.nn as nn

import torch.nn as nn
import torch
import numpy as np
import json
import time
import datetime
import pandas as pd

from torch.utils.data import DataLoader

from os import listdir
from os.path import isfile, join


import torch
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
        
            )
        for m in self.M1:
            if isinstance(m, (nn.Linear)):
                nn.init.normal_(m.weight, mean=0, std=0.02)
        
    def forward(self,X):
        X = self.M1(X)
        return X


## this is a cuda implementation

device = torch.device("cuda")



## copy pasted from Pytorch 
torch.backends.fp32_precision = "tf32"
torch.backends.cudnn.conv.fp32_precision = "tf32"

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True


batch_size = 128
class MNIST_DataLoader(torch.utils.data.Dataset):

    def __init__(self,dir,device):
        self.Data = torch.tensor([])
        self.Labels = torch.tensor([])
        self.testData =torch.tensor([])
        self.testLabel =torch.tensor([])

        df = pd.read_csv('emnist-letters-train.csv') # accessing the csv
        td = df.to_numpy()

        self.Data = torch.tensor(td[:,1:])
        self.Data = self.Data.to(torch.float32)
        self.Labels = torch.tensor(td[:,0])
        #print(self.Data.shape)
        self.Data = (self.Data / 127.5) - 1
    def __len__(self):
        return int((self.Labels.shape[0]//batch_size)*batch_size)
        
    def __getitem__(self,index):
        return self.Data[index] , self.Labels[index]


seed_dim = 100

Data = MNIST_DataLoader("",device)
train_dataloader = DataLoader(Data, batch_size=batch_size, shuffle=True)


Generator = Generator(seedDim=seed_dim).to(device)
Discriminator = Discriminator().to(device)

#Generator = torch.compile(Generator)
#Discriminator= torch.compile(Discriminator)

#criterion = nn.BCELoss()
optimizerD = torch.optim.Adam(Discriminator.parameters(), lr=1e-4, betas=(0.0, 0.9))
optimizerG = torch.optim.Adam(Generator.parameters(), lr=1e-4, betas=(0.0, 0.9))

epochs = 2000
n_critic = 5
w_clip = 0.01
lambdaFactor = 10

lossesG = []
lossesD = []


def gradient_penalty(critic, real, fake, device):
    m = real.size(0)
    eps = torch.rand(m, 1, device=device)          # one ε per sample
    x_hat = (eps * real + (1 - eps) * fake).requires_grad_(True)

    d_hat = critic(x_hat)                            # D(x̂), shape [m,1]
    grads = torch.autograd.grad(
        outputs=d_hat,
        inputs=x_hat,
        grad_outputs=torch.ones_like(d_hat),
        create_graph=True,                           # so we can backprop through it
        retain_graph=True,
    )[0]                                             # shape [m, 784]
    

    grad_norm = grads.norm(2, dim=1)                 # ‖∇‖₂ per sample
    return ((grad_norm - 1) ** 2).mean()



for ep in range(epochs):
    enumerator_data = enumerate(train_dataloader,start=1)
    for i, data in enumerator_data:
        optimizerD.zero_grad()
        #Real_Label = torch.full((data[0].shape[0],1),1,dtype=torch.float32,device=device)
        real_X = data[0].to(device)
        Z = torch.randn((batch_size,seed_dim),device=device)
        output1 = Discriminator(real_X)
        D_x = output1.mean()
        Fakes = Generator(Z).detach()
        output2 = Discriminator(Fakes)
        D_G_z = output2.mean()
        #Loss_D = - (D_x - D_G_z)
        Loss_D = (-(D_x - D_G_z)) + gradient_penalty(Discriminator,real_X,Fakes,device=device)*lambdaFactor # gradient Penalty
        Loss_D.backward()
        optimizerD.step()

        #for p in Discriminator.parameters():   # weight clipping
        #    p.data.clamp_(-w_clip, w_clip)

        if (i+1) % n_critic == 0:
            optimizerG.zero_grad()
            Z = torch.randn((batch_size,seed_dim),device=device)
            Fakes = Generator(Z)
            output = Discriminator(Fakes)
            D_G_z2 = -output.mean()
            Loss_G = D_G_z2
            Loss_G.backward()
            optimizerG.step()

        if (i+1) % 500 == 0:
            lossesG.append(Loss_G.detach())
            lossesD.append(-Loss_D.detach())
            info = f"Epoch {ep} : Step {i} \n: Generator Loss: {Loss_G.detach()} : Discriminator Loss: {-Loss_D.detach()} : \n D_x -> {D_x.detach()} : D_G_z -> {D_G_z.detach()}"
            with open("logGAN.txt","a") as f:
                f.write(info)
            print("==========================")
            print(f"Epoch {ep} : Step {i} \n: Generator Loss: {Loss_G.detach()} : Discriminator Loss: {-Loss_D.detach()} : \n D_x -> {D_x.detach()} : D_G_z -> {D_G_z.detach()}")
            print("==========================")
    
    if ((ep % 25) == 0) and (ep != 0):
        ts = time.time()
        stmp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        torch.save({
            "Epoch":ep,
            "Generator" : Generator.state_dict(),
            "Discriminator" : Discriminator.state_dict(),
            "time"    : stmp,
            "Batch_Size" : batch_size,
            'OptimizerG': optimizerG.state_dict(),
            'OptimizerD': optimizerD.state_dict(),
            'RMSPROP' : True,
            'LossesG':lossesG,
            'LossesD':lossesD,
            'GAN':True
                    }, f'Checkpoint_WGAN_GP.pt')
            
ts = time.time()
stmp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
torch.save({
            "Epoch":ep,
            "Generator" : Generator.state_dict(),
            "Discriminator" : Discriminator.state_dict(),
            "time"    : stmp,
            "Batch_Size" : batch_size,
            'OptimizerG': optimizerG.state_dict(),
            'OptimizerD': optimizerD.state_dict(),
            'RMSPROP' : True,
            'LossesG':lossesG,
            'LossesD':lossesD,
            'GAN':True
                    }, f'Checkpoint_WGAN_GP.pt')




        




