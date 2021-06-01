import torch
import torch.nn as nn 
import torch.nn.functional as f
import numpy as np
class UNET(nn.Module):
    """
    Description: 
    Inputs:
        - Input data (spectrograms)
    Outputs:
        - Reconstructed data
        - Latent space data
    """
    def __init__(self):
        super(UNET, self).__init__()
        
        #down
        self.c11 = nn.Conv2d(1,8, kernel_size=(3,3), padding=(1,1))#, padding="same")
        self.c12 = nn.Conv2d(8,8, kernel_size=(3,3))#, padding="same")
        self.pool1=nn.MaxPool2d((2, 2))
        self.drop1=nn.Dropout(.25)
        
        self.c21 = nn.Conv2d(8,16, kernel_size=(3,3), padding=(1,1))#, padding="same")
        self.c22 = nn.Conv2d(16,16, kernel_size=(3,3))#, padding="same")
        self.pool2=nn.MaxPool2d((2, 2))
        self.drop2=nn.Dropout(.5)
        
        self.c31 = nn.Conv2d(16,32, kernel_size=(3,3), padding=(1,1))#, padding="same")
        self.c32 = nn.Conv2d(32,32, kernel_size=(3,3))#, padding="same")
        self.pool3=nn.MaxPool2d((2, 2))
        self.drop3=nn.Dropout(.5)
        
        self.c41 = nn.Conv2d(32,64, kernel_size=(3,3), padding=(1,1))#, padding="same")
        self.c42= nn.Conv2d(64,64, kernel_size=(3,3))#, padding="same")
        self.pool4=nn.MaxPool2d((2, 2))
        self.drop4=nn.Dropout(.5)
        
        
        #mid
        self.convmid1 = nn.Conv2d(64,128, kernel_size=(3,3), padding=(1,0))#, padding="same")
        self.convmid2 = nn.Conv2d(128 ,128, kernel_size=(1,1))#, padding="same")
        
        
        #need to figure out dimension stuff to get flattened to be dim 9
#         self.l1 = nn.Linear(1152,9)
#         self.f1 = nn.Flatten()
        
#         #up
#         self.l2=nn.Linear(9,1152)
        
        self.dc41=nn.ConvTranspose2d(128, 64, kernel_size=(3,3), padding=(1,0))#, stride=(2,2), padding=(1,0))
        self.ddrop4=nn.Dropout(.5)
        self.dc42=nn.Conv2d(64,64, kernel_size=(3,3))
        self.dc43=nn.Conv2d(64,64, kernel_size=(3,3))
        
        
        self.dc31=nn.ConvTranspose2d(64, 32, kernel_size=(3,3))#,padding=(2,1))#, stride=(2,2), padding=(0,1))      
        self.ddrop3=nn.Dropout(.5)
        self.dc32=nn.Conv2d(32,32, kernel_size=(3,3))
        self.dc33=nn.Conv2d(32,32, kernel_size=(3,3))
        
        self.dc21=nn.ConvTranspose2d(32, 16, kernel_size=(3,3))#, stride=(2,2), padding=(0,1))
        self.ddrop2=nn.Dropout(.5)
        self.dc22=nn.Conv2d(16,16, kernel_size=(3,3))
        self.dc23=nn.Conv2d(16,16, kernel_size=(3,3))
        
        
        self.dc1=nn.ConvTranspose2d(16, 8, kernel_size=(3,3))#, stride=(2,2), padding=(0,0))
        self.ddrop1=nn.Dropout(.5)
        self.dc12=nn.Conv2d(8,8, kernel_size=(3,3))
        self.dc13=nn.Conv2d(8,8, kernel_size=(3,3))
        
        
        #self.dc1=nn.ConvTranspose2d(8, 1, kernel_size=(3,3))#, stride=(2,2), padding=(0,1))
        out=nn.Conv2d(8,1, kernel_size=(1,1))
        

    def forward(self, x):
        
        #down 
        x1 = f.relu(self.c11(x))
        x1 = f.relu(self.c12(x1))
        x1=self.pool1(x1)
        x1=self.drop1(x1)
        
        
        x2 = f.relu(self.c21(x1))
        x2 = f.relu(self.c22(x2))
        x2=self.pool2(x2)
        x2=self.drop2(x2)
        
        x3 = f.relu(self.c31(x2))
        x3 = f.relu(self.c32(x3))
        x3 = self.pool3(x3)
        x3 = self.drop3(x3)
        
        x4 = f.relu(self.c41(x3))
        x4 = f.relu(self.c42(x4))
        x4 = self.pool4(x4)
        x4 = self.drop4(x4)
        #mid
        xmid=self.convmid1(x4)
        xmid=self.convmid2(xmid)
        
#         #latent
# #         x_latent=f.relu(self.l1(self.f1(x5)))
# #         x_unlatent=f.relu(self.l2(x_latent))
# #         #up
# #         x_unlatent =  x_unlatent.view(-1, 128, 3, 3)

        
#         x6= f.relu(self.c6(x_unlatent))
        
        #up
        xd4=self.dc41(xmid)
        print(xd4.shape)
        print(x4.shape)
        xd4=torch.cat([xd4, x4])
        xd4=f.relu(self.dc42(xd4))
        xd4=f.relu(self.dc43(xd4))
        
        xd3=self.dc31(xd4)
        
        print(xd3.shape)
        print(x3.shape)
        xd3=torch.cat([xd3, x3])
        xd3=f.relu(self.dc32(xd3))
        xd3=f.relu(self.dc33(xd3))
        
        xd2=self.dc21(xd3)
        xd2=torch.cat([xd2, x2])
        xd2=f.relu(self.dc22(xd2))
        xd2=f.relu(self.dc23(xd2))
    
    
        xd1=self.dc11(xd2)
        xd1=torch.cat([xd1, x1])
        xd1=f.relu(self.dc12(xd1))
        xd1=f.relu(self.dc13(xd1))
        
        return f.sigmoid(out(xd1))

