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
        self.c1 = nn.Conv2d(1, 8, kernel_size=(3,3), stride=(2,2), padding=(1,1))
        self.a1 = nn.ReLU(True)(self.c1)

        self.c2 = nn.Conv2d(8, 16, kernel_size=(3,3), stride=(2,2), padding=(1,1))(self.a1)
        self.a2 = nn.ReLU(True)(self.c1)

        self.c3 = nn.Conv2d(16, 32, kernel_size=(3,3), stride=(2,2), padding=(1,1))(self.a2)
        self.a3 = nn.ReLU(True)(self.c1)

        self.c4 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=(2,2), padding=(1,1))(self.a3)
        self.a4 = nn.ReLU(True)(self.c1)

        self.c5 = nn.Conv2d(64, 128, kernel_size=(3,3), stride=(2,2), padding=(1,0))(self.a4)

        #up
        self.c6=nn.ConvTranspose2d(128, 64, kernel_size=(3,3), stride=(2,2), padding=(1,0))(self.c5)
        self.a6=nn.ReLU(True)(self.c6)
        self.u6=concatenate([self.a6, self.a4])

        self.c7=nn.ConvTranspose2d(64, 32, kernel_size=(3,3), stride=(2,2), padding=(0,1))(self.u6)
        self.a7=nn.ReLU(True)(self.c7)
        self.u7=concatenate([self.a7, self.a3])

        self.c8=nn.ConvTranspose2d(32, 16, kernel_size=(3,3), stride=(2,2), padding=(0,1))(self.u7)
        self.a8=nn.ReLU(True)(self.c8)
        self.u8=concatenate([self.a8, self.a2])

        self.c9=nn.ConvTranspose2d(16, 8, kernel_size=(3,3), stride=(2,2), padding=(0,0))(self.u8)
        self.a9=nn.ReLU(True)(self.c9)
        self.u9= concatenate([self.a9, self.a1])

        self.c10=nn.ConvTranspose2d(8, 1, kernel_size=(3,3), stride=(2,2), padding=(0,1))(self.u9)

    def forward(self, x):
        x1 = self.contraction1(x)
        x2 = self.contraction2(x1)
        x3 = self.contraction3(x2)
        x4 = self.contraction4(x3)
        x = self.expansion(z)
        return x, z
