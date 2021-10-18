from torch import nn
import torch


class BiasLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.3, normalisation_mode='global'):
        super(BiasLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.norm_mode = normalisation_mode
        self.global_min = 100000

    def norm_global(self, tensor):
        min = tensor.clone().min()
        max = tensor.clone().max()

        if min < self.global_min:
            self.global_min = min
        normalised = ((tensor - self.global_min) / (max - min))
        return normalised

    def norm_local(self, tensor):
        min = tensor.clone().min()
        max = tensor.clone().max()

        normalised = ((tensor - min) / (max - min))

        return normalised

    def forward(self, features, output, target):
        features_copy = features.clone().detach()
        features_dp = features_copy.reshape(features_copy.shape[0], -1)

        features_dp = (torch.var(features_dp, dim=1))
        if self.norm_mode == 'global':
            variance_dp_normalised = self.norm_global(features_dp)
        else:
            variance_dp_normalised = self.norm_local(features_dp)

        weights = ((torch.exp(variance_dp_normalised * self.beta) - 1.) / 1.) + self.alpha
        loss = weights * self.ce(output, target)

        loss = loss.mean()

        return loss

