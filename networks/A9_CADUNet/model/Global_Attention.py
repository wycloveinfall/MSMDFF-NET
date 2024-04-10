# -*-coding:utf-8-*-
import torch
import torch.nn as nn

class global_atention(nn.Module):
    def __init__(self,input_size):
        super(global_atention, self).__init__()

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(input_size,input_size)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(input_size, input_size)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        out = self.global_avg_pool(x)
        out = out.squeeze()
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmod(out)
        out = out.unsqueeze(dim=-1).unsqueeze(dim=-1)
        out = torch.multiply(out,x)
        out = out + x
        return out

if __name__ == '__main__':
    from torchsummary import summary

    x_r = torch.rand([2,2112,16,16]).cuda()
    model = global_atention(2112).cuda()
    model(x_r)