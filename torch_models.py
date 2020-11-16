import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
from torch.autograd import Variable
import math


class NeuralModule(nn.Module):
    def __init__(self):
        super(NeuralModule, self).__init__()
        pass

    def countParameters(self):
        """Function to count parameters in the model"""

        pp = 0
        for p in list(self.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        print('-' * 10)
        print(self)
        print('Num params: ', pp)
        print('-' * 10)
        return pp

    def loadModel(self, model_path):
        self.load_state_dict(torch.load(model_path), strict=False)


class BlockA(NeuralModule):
    def __init__(self, n_input, n_output):
        super(BlockA, self).__init__()
        self.left1 = nn.Conv3d(n_input, n_output, 3, 2)
        self.pad1 = nn.ReplicationPad3d((1, 1, 1, 1, 1, 1))
        self.left2 = nn.Conv3d(n_output, n_output, 3, 1, padding=1)
        self.right = nn.Conv3d(n_input, n_output, 3, 2)
        self.relu = nn.ReLU()


    def forward(self, x):
        l = self.left1(x)
        l = self.relu(l)
        # l = self.pad1(l)
        l = self.left2(l)
        l = self.relu(l)
        r = self.right(x)
        r = self.relu(r)

        return l+r


class BlockA_Transpose(NeuralModule):
    def __init__(self, n_input, n_output):
        super(BlockA_Transpose, self).__init__()
        self.left1 = nn.ConvTranspose3d(n_input, n_output, 3, 1, padding=1)
        self.pad1 = nn.ReplicationPad3d((1, 1, 1, 1, 1, 1))
        self.left2 = nn.ConvTranspose3d(n_output, n_output, 3, 2)
        self.right = nn.ConvTranspose3d(n_input, n_output, 3, 2)
        self.relu = nn.ReLU()


    def forward(self, x):
        l = self.left1(x)
        l = self.relu(l)
        # l = self.pad1(l)
        l = self.left2(l)
        l = self.relu(l)
        r = self.right(x)
        r = self.relu(r)

        # print("ATranspose", x.size(), l.size(), r.size())

        return l+r


class BlockB(NeuralModule):
    def __init__(self, n_input, n_output):
        super(BlockB, self).__init__()
        self.pad1 = nn.ReplicationPad3d((1, 1, 1, 1, 1, 1))
        self.left1 = nn.Conv3d(n_input, n_output, 3, 1, padding=1)
        self.pad2 = nn.ReplicationPad3d((1, 1, 1, 1, 1, 1))
        self.left2 = nn.Conv3d(n_output, n_output, 3, 1, padding=1)
        self.pad3 = nn.ReplicationPad3d((1, 1, 1, 1, 1, 1))
        self.right = nn.Conv3d(n_input, n_output, 3, 1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # l = self.pad1(x)
        l = self.left1(x)
        l = self.relu(l)
        # l = self.pad2(l)
        l = self.left2(l)
        l = self.relu(l)

        # r = self.pad3(x)
        r = self.right(x)
        r = self.relu(r)

        # print(x.size(), l.size(), r.size())

        return l+r


class BlockB_Transpose(NeuralModule):
    def __init__(self, n_input, n_output):
        super(BlockB_Transpose, self).__init__()
        self.pad1 = nn.ReplicationPad3d((1, 1, 1, 1, 1, 1))
        self.left1 = nn.ConvTranspose3d(n_output, n_output, 3, 1, padding=1, output_padding=0)
        self.pad2 = nn.ReplicationPad3d((1, 1, 1, 1, 1, 1))
        self.left2 = nn.ConvTranspose3d(n_input, n_output, 3, 1, padding=1, output_padding=0)
        self.pad3 = nn.ReplicationPad3d((1, 1, 1, 1, 1, 1))
        self.right = nn.ConvTranspose3d(n_input, n_output, 3, 1, padding=1, output_padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        # l = self.pad1(x)
        l = self.left2(x)
        l = self.relu(l)
        # l = self.pad2(l)
        l = self.left1(l)
        l = self.relu(l)

        # r = self.pad3(x)
        r = self.right(x)
        r = self.relu(r)

        # print("BTranspose", x.size(), l.size(), r.size())

        return l+r


class BlockSkip(NeuralModule):
    def __init__(self, n_input, n_output):
        super(BlockSkip, self).__init__()
        self.pad1 = nn.ReplicationPad3d((1, 1, 1, 1, 1, 1))
        self.x = nn.Conv3d(n_input, n_output, 3, 1)
        self.x_bn = nn.BatchNorm3d(n_output, affine=True, track_running_stats=True)
        self.pad2 = nn.ReplicationPad3d((1, 1, 1, 1, 1, 1))
        self.y = nn.Conv3d(n_output, n_output, 3, 1)
        self.y_bn = nn.BatchNorm3d(n_output, affine=True, track_running_stats=True)
        self.relu = nn.ReLU()
        self.all_layers = nn.Sequential(*[self.pad1, self.x, self.x_bn, self.relu, self.pad2, self.y, self.y_bn, self.relu])

    def forward(self, x):
        return self.all_layers(x)


class BlockSkip_Transpose(NeuralModule):
    def __init__(self, n_input, n_output):
        super(BlockSkip_Transpose, self).__init__()
        self.pad1 = nn.ReplicationPad3d((1, 1, 1, 1, 1, 1))
        self.x = nn.ConvTranspose3d(n_input, n_output, 3, 1, padding=1)
        self.x_bn = nn.BatchNorm3d(n_output, affine=True, track_running_stats=True)
        self.pad2 = nn.ReplicationPad3d((1, 1, 1, 1, 1, 1))
        self.y = nn.ConvTranspose3d(n_output, n_output, 3, 1, padding=1)
        self.y_bn = nn.BatchNorm3d(n_output, affine=True, track_running_stats=True)
        self.relu = nn.ReLU()
        self.all_layers = nn.Sequential(*[# self.pad2,
                                          self.x,
                                          self.x_bn,
                                          self.relu,
                                          # self.pad1,
                                          self.y,
                                          self.y_bn,
                                          self.relu])

    def forward(self, x):
        # print("SkipTranspose", x.size())
        return self.all_layers(x)


class JianPei(NeuralModule):
    def __init__(self):
        super(JianPei, self).__init__()

        self.blocks = ['A', 'B', 'S', 'A', 'B', 'B', 'S', 'A', 'B', 'S']
        self.filters = [32, 48, 48, 64, 96, 96, 96, 128, 128, 128]
        in_filters = 1
        arch_list = []
        for b, f in zip(self.blocks, self.filters):
            if b == 'A':
                arch_list.append(BlockA(in_filters, f))
                in_filters = f
            elif b == 'B':
                arch_list.append(BlockB(in_filters, f))
                in_filters = f
            elif b == 'S':
                arch_list.append(BlockSkip(in_filters, f))
                in_filters = f
        self.avg = nn.MaxPool3d(4, 4, return_indices=False)
        arch_list.append(self.avg)
        self.fc = nn.Conv3d(in_filters, 2, 1, 1)
        # arch_list.append(self.fc)
        self.forward_layers = nn.Sequential(*arch_list)
        # print(self.forward_layers)

        # self.rev_blocks = self.blocks[::-1]
        # self.rev_filters = self.filters[::-1][1:] + [1]
        # print("Rev filters", self.rev_filters)
        # in_filters = self.rev_filters[0]
        # arch_list = []
        # avg = nn.MaxUnpool3d(4, 4)
        # arch_list.append(avg)
        # for b, f in zip(self.rev_blocks, self.rev_filters):
        #     if b == 'A':
        #         arch_list.append(BlockA_Transpose(in_filters, f))
        #         in_filters = f
        #     elif b == 'B':
        #         arch_list.append(BlockB_Transpose(in_filters, f))
        #         in_filters = f
        #     elif b == 'S':
        #         arch_list.append(BlockSkip_Transpose(in_filters, f))
        #         in_filters = f
        # self.backward_layers = nn.Sequential(*arch_list)
        # print(self.backward_layers)

    def forward(self, x):
        # for l in self.forward_layers:
        #     x = l(x)
        #     print("Output", x.size())
        mid = self.forward_layers(x) # .squeeze(-1).squeeze(-1).squeeze(-1)
        # mid, unpool = self.avg(mid)
        # x = self.backward_layers(mid)
        # print(mid.size(), unpool.size())
        # print("VECTOR SIZE:", mid.size())
        # x = mid
        # for l in self.backward_layers:
        #     if isinstance(l, nn.MaxUnpool3d) == False:
        #         x = l(x)
        #         print("Output", x.size())
        #     else:
        #         x = l(x, unpool)
        #         print("Unpool output", x.size())
        # # x = x.squeeze(-1).squeeze(-1).squeeze(-1)
        return self.fc(mid).squeeze(-1).squeeze(-1).squeeze(-1) # , x

if __name__ == "__main__":
    m = JianPei()
    x = torch.zeros(1, 1, 40, 40, 40)
    vector = m(x) # , reconstruct = m(x)
    print(vector.size()) # , reconstruct.size(), x.size())

