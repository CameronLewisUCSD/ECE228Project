#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class Contraction(nn.Module):
    """
    Description: 
    Inputs:
        - Input data (spectrograms)
    Outputs:
        - Latent feature space data
    """
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            #c1
            nn.Conv2d(1, 8, kernel_size=(3,3), stride=(2,2), padding=(1,1)),
            nn.ReLU(True),
            #c2
            nn.Conv2d(8, 16, kernel_size=(3,3), stride=(2,2), padding=(1,1)),
            nn.ReLU(True),
            #c3
            nn.Conv2d(16, 32, kernel_size=(3,3), stride=(2,2), padding=(1,1)),
            nn.ReLU(True),
            #c3
            nn.Conv2d(32, 64, kernel_size=(3,3), stride=(2,2), padding=(1,1)),
            nn.ReLU(True),
            #c4
            nn.Conv2d(64, 128, kernel_size=(3,3), stride=(2,2), padding=(1,0)),
            nn.ReLU(True),
            
            nn.Flatten(),
            nn.Linear(1152, 9),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x
    
class Expansion(nn.Module):
    """
    Description: 
    Inputs:
        - Latent space data
    Outputs:
        - Reconstructed data
    """
    def __init__(self):
        super(Decoder, self).__init__()
        self.latent2dec = nn.Sequential(
            nn.Linear(9, 1152),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            #u6
            nn.ConvTranspose2d(128, 64, kernel_size=(3,3), stride=(2,2), padding=(1,0)), # <---- Experimental
            nn.ReLU(True),  # <---- Experimental
            #u7
            nn.ConvTranspose2d(64, 32, kernel_size=(3,3), stride=(2,2), padding=(0,1)),
            nn.ReLU(True),
            #u8
            nn.ConvTranspose2d(32, 16, kernel_size=(3,3), stride=(2,2), padding=(0,1)),
            nn.ReLU(True),
            #u9
            nn.ConvTranspose2d(16, 8, kernel_size=(3,3), stride=(2,2), padding=(0,0)),
            nn.ReLU(True),
            #u10
            nn.ConvTranspose2d(8, 1, kernel_size=(3,3), stride=(2,2), padding=(0,1)),
        )

    def forward(self, x):
        x = self.latent2dec(x)
        x = x.view(-1, 128, 3, 3)
        x = self.decoder(x)
        return x[:,:,4:-4,1:]

class UNET(nn.Module):
    """
    Description: Autoencoder model; combines encoder and decoder layers.
    Inputs:
        - Input data (spectrograms)
    Outputs:
        - Reconstructed data
        - Latent space data
    """
    def __init__(self):
        super(AEC, self).__init__()
        self.contraction = Contraction()
        self.expansion = Expansion()

    def forward(self, x):
        z = self.contraction(x)
        x = self.expansion(z)
        return x, z

