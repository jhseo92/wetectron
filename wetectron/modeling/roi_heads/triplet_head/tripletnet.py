import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

class TripletNet(nn.Module):

    def __init__(self, num_classes=21):
        super(TripletNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.batchnorm = nn.BatchNorm2d(128)
        self.classifier = nn.Sequential(
            #nn.Dropout(),
            #nn.Linear(128 * 6 * 6, 2048),
            #nn.ReLU(inplace=True),
            #nn.Dropout(),
            #nn.Linear(2048, 1024),
            #nn.ReLU(inplace=True),
            #nn.Linear(1024, num_classes),
            nn.Linear(128*6*6,2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,num_classes),
        )

    def forward(self, x):
        features = self.features(x)
        x = self.avgpool(features)
        #features = self.batchnorm(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x, features
