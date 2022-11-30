import torch
from torch import nn

# Initialization
def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

# ANN Layer
class Layer(torch.nn.Module):
    def __init__(self, in_dim, h_dim):
        super(Layer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.LeakyReLU(0.2, inplace=True))

        self.linear.apply(xavier_init)

    def forward(self, x):
        return self.linear(x)

class ANN(nn.Module):
    def __init__(self, in_hidden_list):
        super(ANN, self).__init__()
        self.Layer_List = nn.ModuleList(
            [Layer(in_hidden, in_hidden_list[i + 1]) for i, in_hidden in enumerate(in_hidden_list[:-1])])

        self.classifier = nn.Sequential(
            nn.Linear(in_features=in_hidden_list[-1], out_features=1),
            nn.Sigmoid()
        )

        self.embedding_num = len(in_hidden_list) - 1

    def forward(self, x):
        f_ = dict()
        for num in range(self.embedding_num):
            if num == 0:
                f_[num] = self.Layer_List[num](x)
            else:
                f_[num] = self.Layer_List[num](f_[num - 1])

        output = self.classifier(f_[num])

        return output