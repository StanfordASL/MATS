import torch
import torch.distributions as td


class GMM2D(td.MixtureSameFamily):
    def __init__(self, mixture_distribution, component_distribution):
        super(GMM2D, self).__init__(mixture_distribution, component_distribution)

    def mode_mode(self):
        mode_k = torch.argmax(self.mixture_distribution.probs[0, 0]).item()
        mode_gaussian = self.component_distribution.mean[:, 0, mode_k, :2]
        return mode_gaussian

    def position_log_prob(self, x):
        # Computing the log probability over only the positions.
        component_dist = td.MultivariateNormal(loc=self.component_distribution.mean[..., :2],
                                               scale_tril=self.component_distribution.scale_tril[..., :2, :2])
        position_dist = td.MixtureSameFamily(self.mixture_distribution, component_dist)
        return position_dist.log_prob(x)

    @property
    def pis(self):
        return self.mixture_distribution.probs[0, 0]
