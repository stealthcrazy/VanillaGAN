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

import vanillaGAN as GAN


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


Generator = GAN.Generator(seedDim=seed_dim).to(device)
Discriminator = GAN.Discriminator().to(device)

#Generator = torch.compile(Generator)
#Discriminator= torch.compile(Discriminator)

criterion = nn.BCELoss()
optimizerD = torch.optim.Adam(Discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(Generator.parameters(), lr=2.5e-4, betas=(0.5, 0.999))

epochs = 100

lossesG = []
lossesD = []


for ep in range(epochs):

    for i, data in enumerate(train_dataloader):
        Real_Label = torch.full((data[0].shape[0],1),1,dtype=torch.float32,device=device)
        Fake_Label = torch.full((data[0].shape[0],1),0,dtype=torch.float32,device=device)


        optimizerD.zero_grad()
        real_X = data[0].to(device)
    
        output = Discriminator(real_X)
        Loss_D_Real = criterion(output,Real_Label)
        D_x = output.mean().detach()
        
        Z = torch.randn((batch_size,seed_dim),device=device)
        Fakes = Generator(Z)
    

        output = Discriminator(Fakes.detach())
        D_G_z = output.mean().detach()
        Loss_D_Fake = criterion(output,Fake_Label)

        Loss_D = Loss_D_Fake + Loss_D_Real
        Loss_D.backward()

        optimizerD.step()

        optimizerG.zero_grad()

        Z = torch.randn((batch_size,seed_dim),device=device)
        Fakes = Generator(Z)
        output = Discriminator(Fakes)
        D_G_z2 = output.mean().detach()
        Loss_G = criterion(output,Real_Label)
        Loss_G.backward()
        optimizerG.step()
        if i % 100 == 0:
            lossesG.append(Loss_G.detach())
            lossesD.append(Loss_D.detach())
            info = f"Epoch {ep} : Step {i} \n: Generator Loss: {Loss_G.detach()} : Discriminator Loss: {Loss_D.detach()} : \n D_x -> {D_x} : D_G_z -> {D_G_z} : D_G_z2 -> {D_G_z2}"
            with open("logGAN.txt","a") as f:
                f.write(info)
            print("==========================")
            print(f"Epoch {ep} : Step {i} \n: Generator Loss: {Loss_G} : Discriminator Loss: {Loss_D} : \n D_x -> {D_x} : D_G_z -> {D_G_z} : D_G_z2 -> {D_G_z2}")
            print("==========================")
    
    if ((ep % 50) == 0) and (ep != 0):
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
            'BCE' : True,
            'ADAM' : True,
            'LossesG':lossesG,
            'LossesD':lossesD,
            'GAN':True
                    }, f'Checkpoint_Meta.pt')
            
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
            'BCE' : True,
            'ADAM' : True,
            #'LossesG':lossesG,
            #'LossesD':lossesD,
            'GAN':True
                    }, f'Checkpoint_Meta.pt')




        




