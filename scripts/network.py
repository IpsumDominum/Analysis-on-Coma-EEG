'''
---------------------
Plan of attack:
First do a raw CNN classification thing.
Then do some latent variable feature extraction thing.
'''
import torch.nn as nn
import torch.nn.functional as F

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel,self).__init__()
        self.conv1 = nn.Conv1d(22,20,5000)
        self.conv2 = nn.Conv1d(20,40,5000)
        self.conv3 = nn.Conv1d(40,80,3000)
        self.linear = nn.Linear(80*253,253)
        self.linear2 = nn.Linear(253,1)
    def forward(self,x):
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = F.tanh(self.conv3(x))
        x = x.flatten()
        x = F.tanh(self.linear(x))
        x = F.tanh(self.linear2(x))
        return x
