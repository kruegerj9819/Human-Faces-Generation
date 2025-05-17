import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_size=128):
        super(Generator, self).__init__()

        self.in_to_h1 = self.scaling(latent_size, 512, padding=0)
        self.h1_to_h2 = self.scaling(512, 256)
        self.h2_to_h3 = self.scaling(256, 128)
        self.h3_to_h4 = self.scaling(128, 64)
        self.h4_to_out = self.scaling(64, 3, last_layer=True)

    def scaling(self, in_channels, out_channels, padding=1, last_layer=False):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, padding, bias=False)
        ]
        if not last_layer:
            layers += [
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ]
        else:
            layers += [
                nn.Tanh()
            ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.in_to_h1(x) # 512 x 4 x 4
        x = self.h1_to_h2(x) # 256 x 8 x 8
        x = self.h2_to_h3(x) # 128 x 16 x 16
        x = self.h3_to_h4(x) # 64 x 32 x 32
        return self.h4_to_out(x) # 3 x 64 x 64

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.in_to_h1 = self.scaling(3, 64)
        self.h1_to_h2 = self.scaling(64, 128)
        self.h2_to_h3 = self.scaling(128, 256)
        self.h3_to_h4 = self.scaling(256, 512)
        self.h4_to_out = self.scaling(512, 1, padding=0)

    def scaling(self, in_channels, out_channels, padding=1):
        layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, padding),
            nn.LeakyReLU(inplace=True)
        )
        return layers

    def forward(self, x):
        x = self.in_to_h1(x) # 64 x 32 x 32
        x = self.h1_to_h2(x) # 128 x 16 x 16
        x = self.h2_to_h3(x) # 256 x 8 x 8
        x = self.h3_to_h4(x) # 512 x 4 x 4
        return self.h4_to_out(x).view(-1, 1).squeeze(1)