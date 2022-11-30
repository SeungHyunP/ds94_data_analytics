import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.parameter import Parameter

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

# SoftTriple
class SoftTripleLoss(nn.Module):
    def __init__(self, dim, device):
        super(SoftTripleLoss, self).__init__()
        self.device = device
        self.tau = 0.1
        self.dim = dim
        self.margin = 1e-2
        self.cN = 2
        self.K = 10
        self.fc = Parameter(torch.Tensor(self.dim, self.cN * self.K))
        self.weight = torch.zeros(self.cN * self.K, self.cN * self.K, dtype=torch.bool).to(device)
        for i in range(0, self.cN):
            for j in range(0, self.K):
                self.weight[i * self.K + j, i * self.K + j + 1:(i + 1) * self.K] = 1

        # Weight Initialization
        init.kaiming_uniform_(self.fc)

    def forward(self, input, target):
        # Soft Triple Loss
        input = F.normalize(input, p=2, dim=1)
        centers = F.normalize(self.fc, p=2, dim=0)
        simInd = input.matmul(centers)
        simStruc = simInd.reshape(-1, self.cN, self.K)
        prob = F.softmax(simStruc, dim=2)
        simClass = torch.sum(prob * simStruc, dim=2)
        marginM = torch.zeros(simClass.shape).to(self.device)
        marginM[torch.arange(0, marginM.shape[0]), target.detach().cpu().numpy()] = self.margin
        lossClassify = F.cross_entropy((simClass-marginM), target.long())

        # Regularization Term
        simCenter = centers.t().matmul(centers)
        reg = torch.sum(torch.sqrt(2.0 + 1e-5 - 2. * simCenter[self.weight])) / (self.cN * self.K * (self.K - 1.))

        loss = lossClassify+self.tau*reg

        return loss

    def predict(self, input):
        # Soft Triple Loss
        input = F.normalize(input, p=2, dim=1)
        centers = F.normalize(self.fc, p=2, dim=0)
        simInd = input.matmul(centers)
        simStruc = simInd.reshape(-1, self.cN, self.K)
        prob = F.softmax(simStruc, dim=2)
        simClass = torch.sum(prob * simStruc, dim=2)

        # Return Object =>  Prediction // Cluster 1, 2
        predict_prob = torch.softmax(simClass, dim=1)
        prediction = torch.argmax(predict_prob, dim=1)

        # If Cluster
        cluster_0 = torch.argmax(prob[prediction == 0, 0, :], dim=1)
        cluster_1 = torch.argmax(prob[prediction == 1, 1, :], dim=1) + self.K

        # Class-1 Probability, Cluster, Cluster
        return predict_prob[:, 1], cluster_0, cluster_1

class Soft_Triple(nn.Module):
    def __init__(self, in_hidden_list, device):
        super(Soft_Triple, self).__init__()
        self.Layer_List = nn.ModuleList(
            [Layer(in_hidden, in_hidden_list[i + 1]) for i, in_hidden in enumerate(in_hidden_list[:-1])])

        self.classifier = SoftTripleLoss(in_hidden_list[-1], device).to(device)

        self.embedding_num = len(in_hidden_list) - 1

    def forward(self, x, y):
        f_ = dict()
        for num in range(self.embedding_num):
            if num == 0:
                f_[num] = self.Layer_List[num](x)
            else:
                f_[num] = self.Layer_List[num](f_[num - 1])

        output = self.classifier(f_[num], y)

        return output

    def predict(self, x):
        f_ = dict()
        for num in range(self.embedding_num):
            if num == 0:
                f_[num] = self.Layer_List[num](x)
            else:
                f_[num] = self.Layer_List[num](f_[num - 1])

        prob, cluster_0, cluster_1 = self.classifier.predict(f_[num])
        return prob, cluster_0, cluster_1