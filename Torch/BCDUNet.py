import torch
import torch.nn as nn
import numpy as np
from ConvLSTM import ConvBLSTM, ConvLSTM

class BCDUNet(nn.Module):
    def __init__(self, input_dim=3, output_dim=3, num_filter=64, frame_size=(256, 256), bidirectional=False, norm='instance'):
        super(BCDUNet, self).__init__()
        self.num_filter = num_filter
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.frame_size = np.array(frame_size)

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, num_filter, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(num_filter, num_filter, kernel_size=3, stride=1, padding=1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(num_filter, num_filter*2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(num_filter*2, num_filter*2, kernel_size=3, stride=1, padding=1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(num_filter*2, num_filter*4, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(num_filter*4, num_filter*4, kernel_size=3, stride=1, padding=1)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(num_filter*4, num_filter*8, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(num_filter*8, num_filter*8, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.5),
            nn.Conv2d(num_filter*8, num_filter*8, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(num_filter*8, num_filter*8, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.5),
            nn.Conv2d(num_filter*16, num_filter*8, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(num_filter*8, num_filter*8, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.5)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(num_filter*2, num_filter*4, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(num_filter*4, num_filter*4, kernel_size=3, stride=1, padding=1)
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(num_filter, num_filter*2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(num_filter*2, num_filter*2, kernel_size=3, stride=1, padding=1)
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(num_filter//2, num_filter, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(num_filter, num_filter, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(num_filter, num_filter//2, kernel_size=3, stride=1, padding=1)
        )

        self.conv9 = nn.Conv2d(num_filter//2, output_dim, kernel_size=1, stride=1)

        self.convt1 = nn.Sequential(
            nn.ConvTranspose2d(num_filter*8, num_filter*4, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(num_filter*4),
            nn.ReLU(inplace=True)
        )

        self.convt2 = nn.Sequential(
            nn.ConvTranspose2d(num_filter*4, num_filter*2, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(num_filter*2),
            nn.ReLU(inplace=True)
        )

        self.convt3 = nn.Sequential(
            nn.ConvTranspose2d(num_filter*2, num_filter, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(num_filter),
            nn.ReLU(inplace=True)
        )

        if bidirectional:
            self.clstm1 = ConvBLSTM(num_filter*4, num_filter*2, (3, 3), (1,1), 'tanh', list(self.frame_size//4))
            self.clstm2 = ConvBLSTM(num_filter*2, num_filter, (3, 3), (1,1), 'tanh', list(self.frame_size//2))
            self.clstm3 = ConvBLSTM(num_filter, num_filter//2, (3, 3), (1,1), 'tanh', list(self.frame_size))
        else:
            self.clstm1 = ConvLSTM(num_filter*4, num_filter*2, (3, 3), (1,1), 'tanh', list(self.frame_size//4))
            self.clstm2 = ConvLSTM(num_filter*2, num_filter, (3, 3), (1,1), 'tanh', list(self.frame_size//2))
            self.clstm3 = ConvLSTM(num_filter, num_filter//2, (3, 3), (1,1), 'tanh', list(self.frame_size))
        
    def forward(self, x):
        N = self.frame_size
        conv1 = self.conv1(x)
        pool1 = self.maxpool(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.maxpool(conv2)
        conv3 = self.conv3(pool2)
        drop3 = self.dropout(conv3)
        pool3 = self.maxpool(conv3)
        drop4_3 = self.conv4(pool3)

        up6 = self.convt1(drop4_3)
        x1 = drop3.view(-1,1,self.num_filter*4,*(N//4))
        x2 = up6.view(-1,1,self.num_filter*4,*(N//4))
        merge6 = torch.cat((x1, x2), 1)
        merge6 = self.clstm1(merge6)
        conv6 = self.conv6(merge6)

        up7 = self.convt2(conv6)
        x1 = conv2.view(-1,1,self.num_filter*2,*(N//2))
        x2 = up7.view(-1,1,self.num_filter*2,*(N//2))
        merge7 = torch.cat((x1, x2), 1)
        merge7 = self.clstm2(merge7)
        conv7 = self.conv7(merge7)

        up8 = self.convt3(conv7)
        x1 = conv1.view(-1,1,self.num_filter,*N)
        x2 = up8.view(-1,1,self.num_filter,*N)
        merge8 = torch.cat((x1, x2), 1)
        merge8 = self.clstm3(merge8)
        conv8 = self.conv8(merge8)
        conv9 = self.conv9(conv8)

        return conv9
