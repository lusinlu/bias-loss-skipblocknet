from torch import nn
import torch

GLOBAL_MIN = 100000


class BiasLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.3):
        super(BiasLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce = nn.CrossEntropyLoss(reduction='none')


    def normalise(self, tensor):
        min = tensor.clone().min()

        max = tensor.clone().max()
        global GLOBAL_MIN
        if min < GLOBAL_MIN:
            GLOBAL_MIN = min
        normalised = ((tensor - GLOBAL_MIN) / (max - min))
        return normalised

    def forward(self, features, output, target):
        features_copy = features.clone().detach()
        features_per_sample = features_copy.reshape(features_copy.shape[0], -1)

        variance_per_sample = (torch.var(features_per_sample, dim=1))
        variance_per_sample_normalised = self.normalise(variance_per_sample)

        weights = ((torch.exp(variance_per_sample_normalised * self.beta) - 1.) / 1.) + self.alpha
        loss = weights * self.ce(output, target)
        loss = loss.mean()

        return loss

