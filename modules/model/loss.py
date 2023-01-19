import torch
from torch import nn
import torch.nn.functional as F


class ContentLoss(nn.Module):

    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        self.loss = None
        self.target = target.detach()

    def forward(self, _input):
        self.loss = F.mse_loss(_input, self.target)
        return _input


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.loss = None
        self.target = self.gram_matrix(target_feature).detach()

    def forward(self, _input):
        self.loss = F.mse_loss(self.gram_matrix(_input), self.target)
        return _input

    @staticmethod
    def gram_matrix(_input):
        a, b, c, d = _input.size()

        features = _input.view(a * b, c * d)

        return torch.mm(features, features.t()).div(a * b * c * d)


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std
