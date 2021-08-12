import torch
import torch.nn as nn
import torchvision.models as models


class AlexNet(nn.Module):
    def __init__(self, out_features):
        super(AlexNet, self).__init__()
        #model = models.alexnet(pretrained=True)
        model = models.alexnet(pretrained=False)
       
        self.features = nn.Sequential(
            *list(model.children())[:-1]
        )
        
        self.regression = nn.Sequential(
            nn.Dropout(),
            nn.Linear(9216, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, out_features)
        )

    def forward(self, x):
        y = self.features(x)
        y = y.view(y.shape[0], -1)
        y = self.regression(y)
        return y
