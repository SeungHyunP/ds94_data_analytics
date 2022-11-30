import math
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.parameter import Parameter

eps = 1e-7

################# For Loss Function #################
class ContrastMemory(nn.Module):
    """
    memory buffer that supplies large amount of negative samples.
    """
    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5):
        super(ContrastMemory, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K

        self.register_buffer('params', torch.tensor([K, T, -1, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory_v1', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_v2', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, v1, v2, y, idx=None):

        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_v1 = self.params[2].item()
        Z_v2 = self.params[3].item()

        momentum = self.params[4].item()
        batchSize = v1.size(0)
        outputSize = self.memory_v1.size(0)
        inputSize = self.memory_v1.size(1)

        # original score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)

        # sample
        weight_v1 = torch.index_select(self.memory_v1, 0, idx.view(-1)).detach()
        weight_v1 = weight_v1.view(batchSize, K + 1, inputSize)
        out_v2 = torch.bmm(weight_v1, v2.view(batchSize, inputSize, 1))
        # out_v2 = torch.bmm(weight_v1.view(batchSize, inputSize, 1), v2.view(batchSize, inputSize, 1))
        out_v2 = torch.exp(torch.div(out_v2, T))

        # sample
        weight_v2 = torch.index_select(self.memory_v2, 0, idx.view(-1)).detach()
        weight_v2 = weight_v2.view(batchSize, K + 1, inputSize)
        out_v1 = torch.bmm(weight_v2, v1.view(batchSize, inputSize, 1))
        # out_v1 = torch.bmm(weight_v2.view(batchSize, inputSize, 1), v1.view(batchSize, inputSize, 1))
        out_v1 = torch.exp(torch.div(out_v1, T))

        # set Z if haven't been set yet
        if Z_v1 < 0:
            self.params[2] = out_v1.mean() * outputSize
            Z_v1 = self.params[2].clone().detach().item()
            print("normalization constant Z_v1 is set to {:.1f}".format(Z_v1))
        if Z_v2 < 0:
            self.params[3] = out_v2.mean() * outputSize
            Z_v2 = self.params[3].clone().detach().item()
            print("normalization constant Z_v2 is set to {:.1f}".format(Z_v2))

        # compute out_v1, out_v2
        out_v1 = torch.div(out_v1, Z_v1).contiguous()
        out_v2 = torch.div(out_v2, Z_v2).contiguous()

        # update memory
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_v1, 0, y.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(v1, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v1 = l_pos.div(l_norm)
            self.memory_v1.index_copy_(0, y, updated_v1)

            ab_pos = torch.index_select(self.memory_v2, 0, y.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(v2, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v2 = ab_pos.div(ab_norm)
            self.memory_v2.index_copy_(0, y, updated_v2)

        return out_v1, out_v2

class AliasMethod(object):
    """
    From: https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    """
    def __init__(self, probs):

        if probs.sum() > 1:
            probs.div_(probs.sum())
        K = len(probs)
        self.prob = torch.zeros(K)
        self.alias = torch.LongTensor([0]*K)

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.prob[kk] = K*prob
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.alias[small] = large
            self.prob[large] = (self.prob[large] - 1.0) + self.prob[small]

            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller+larger:
            self.prob[last_one] = 1

    def cuda(self):
        self.prob = self.prob.cuda()
        self.alias = self.alias.cuda()

    def draw(self, N):
        """ Draw N samples from multinomial """
        K = self.alias.size(0)

        kk = torch.zeros(N, dtype=torch.long, device=self.prob.device).random_(0, K)
        prob = self.prob.index_select(0, kk)
        alias = self.alias.index_select(0, kk)
        # b is whether a random number is greater than q
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1-b).long())

        return oq + oj

################# Loss Function #################
class CRDLoss(nn.Module):
    """CRD Loss function
    includes two symmetric parts:
    (a) using teacher as anchor, choose positive and negatives over the student side
    (b) using student as anchor, choose positive and negatives over the teacher side
    Args:
        opt.s_dim: the dimension of student's feature
        opt.t_dim: the dimension of teacher's feature
        opt.feat_dim: the dimension of the projection space
        opt.nce_k: number of negatives paired with each positive
        opt.nce_t: the temperature
        opt.nce_m: the momentum for updating the memory buffer
        opt.n_data: the number of samples in the training set, therefor the memory buffer is: opt.n_data x opt.feat_dim
    """
    def __init__(self, opt):
        super(CRDLoss, self).__init__()
        self.embed_s = Embed(opt['s_dim'], opt['feat_dim'])
        self.embed_t = Embed(opt['t_dim'], opt['feat_dim'])
        self.contrast = ContrastMemory(opt['feat_dim'], opt['n_data'], opt['nce_k'], opt['nce_t'], opt['nce_m'])
        self.criterion_t = ContrastLoss(opt['n_data'])
        self.criterion_s = ContrastLoss(opt['n_data'])

    def forward(self, f_s, f_t, idx, contrast_idx=None):
        """
        Args:
            f_s: the feature of student network, size [batch_size, s_dim]
            f_t: the feature of teacher network, size [batch_size, t_dim]
            idx: the indices of these positive samples in the dataset, size [batch_size]
            contrast_idx: the indices of negative samples, size [batch_size, nce_k]
        Returns:
            The contrastive loss
        """
        f_s = self.embed_s(f_s)
        f_t = self.embed_t(f_t)
        out_s, out_t = self.contrast(f_s, f_t, idx, contrast_idx)
        s_loss = self.criterion_s(out_s)
        t_loss = self.criterion_t(out_t)
        loss = s_loss + t_loss
        return loss

class ContrastLoss(nn.Module):
    """
    contrastive loss, corresponding to Eq (18)
    """
    def __init__(self, n_data):
        super(ContrastLoss, self).__init__()
        self.n_data = n_data

    def forward(self, x):
        bsz = x.shape[0]
        m = x.size(1) - 1

        # noise distribution
        Pn = 1 / float(self.n_data)

        # loss for positive pair
        P_pos = x.select(1, 0)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()

        # loss for K negative pair
        P_neg = x.narrow(1, 1, m)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()

        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz

        return loss

class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x


class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss

################# Model #################
def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

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

    def forward(self, x, y, is_feat=False):
        f_ = dict()
        f_list = []
        for num in range(self.embedding_num):
            if num == 0:
                f_[num] = self.Layer_List[num](x)
            else:
                f_[num] = self.Layer_List[num](f_[num - 1])
            f_list.append(f_[num])

        loss = self.classifier(f_[num], y)

        if is_feat:
            return f_list, loss
        else:
            return loss

    def predict(self, x):
        f_ = dict()
        for num in range(self.embedding_num):
            if num == 0:
                f_[num] = self.Layer_List[num](x)
            else:
                f_[num] = self.Layer_List[num](f_[num - 1])

        prob, cluster_0, cluster_1 = self.classifier.predict(f_[num])
        return prob, cluster_0, cluster_1