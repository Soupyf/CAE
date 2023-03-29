import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, (31, 1))  # 120*385 => 58*191
        self.relu1 = nn.Sigmoid()
        # self.conv2 = nn.Conv2d(3, 3, (21, 1))  # 58*191 => 27*94
        # self.relu2 = nn.Sigmoid()
        self.conv3 = nn.Conv2d(3, 2, (31, 1))  # 27*94 => 12*45
        self.batchnorm = nn.BatchNorm2d(2)
        self.relu3 = nn.Sigmoid()
        self.conv4 = nn.Conv2d(2, 2, (9, 1))
        self.relu4 = nn.Sigmoid()
        # self.conv5 = nn.Conv2d(2, 2, (10, 1))
        # self.relu5 = nn.Sigmoid()
        # self.conv6 = nn.Conv2d(2, 2, (8, 1))
        # self.relu6 = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        # x = self.conv2(x)
        # x = self.relu2(x)
        x = self.conv3(x)
        x = self.batchnorm(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        # x = self.conv5(x)
        # x = self.relu5(x)
        # x = self.conv6(x).detach()
        # x = self.relu6(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(2, 2, (9, 1))
        self.relu1 = nn.Sigmoid()
        # self.conv2 = nn.ConvTranspose2d(2, 2, (21, 1))
        # self.relu2 = nn.Sigmoid()
        self.conv3 = nn.ConvTranspose2d(2, 2, (31, 1))
        self.batchnorm = nn.BatchNorm2d(2)
        self.relu3 = nn.Sigmoid()
        self.conv4 = nn.ConvTranspose2d(2, 3, (31, 1))  # 27*93 => 57*189
        self.relu4 = nn.Sigmoid()
        # self.conv5 = nn.ConvTranspose2d(3, 3, (14, 1))  # 57*189 => 120*385
        # self.relu5 = nn.Sigmoid()
        # self.conv6 = nn.ConvTranspose2d(3, 3, (14, 1))
        # self.relu6 = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        # x = self.conv2(x)
        # x = self.relu2(x)
        x = self.conv3(x)
        x = self.batchnorm(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        # x = self.conv5(x)
        # x = self.relu5(x)
        # x = self.conv6(x)
        # x = self.relu6(x)
        return x
