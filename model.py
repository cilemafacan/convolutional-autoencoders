from torch import nn

class CNN(nn.Module):
    def __init__(self, in_channels=3, ch=[16,32,64,128,256]):
        super(CNN, self).__init__()

        self.encoder = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1),
                nn.PReLU(),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
                nn.PReLU(),
                nn.AvgPool2d(kernel_size=2),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
                nn.PReLU(),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
                nn.PReLU(),
                nn.AvgPool2d(kernel_size=2),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
                nn.PReLU(),
                nn.AvgPool2d(kernel_size=2),
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1),
                nn.PReLU(),
                nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1),
                nn.PReLU(),)
        self.flatten = nn.Flatten()
        self.bottleneck = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.PReLU(),
            nn.Linear(in_features=512, out_features=1024),
            nn.PReLU(),
        )
                
        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1),
                nn.PReLU(),
                nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=1),
                nn.PReLU(),
                nn.Upsample(scale_factor=2),
                nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1),
                nn.PReLU(),
                nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=1),
                nn.PReLU(),
                nn.Upsample(scale_factor=2),
                nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1),
                nn.PReLU(),
                nn.Upsample(scale_factor=2),
                nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=1),
                nn.PReLU(),
                nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=3, stride=1))
        
    def forward(self, x):
        encoder = self.encoder(x)
        flatten = self.flatten(encoder)
        bottleneck = self.bottleneck(flatten).view(-1,1024,1,1)
        decoder = self.decoder(bottleneck)
        sigmoid    = nn.Sigmoid()
        output     = sigmoid(decoder)

        return output