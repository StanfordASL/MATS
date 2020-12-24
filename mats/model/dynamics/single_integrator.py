import torch
from model.dynamics import Dynamic


class SingleIntegrator(Dynamic):
    def init_constants(self, batch_size, num_components):
        self.A = torch.eye(2, device=self.device, dtype=torch.float32)
        self.B = torch.diag(torch.tensor([self.dt, self.dt], device=self.device, dtype=torch.float32))

        if batch_size is not None and num_components is not None:
            self.A = self.A.expand(batch_size + (num_components, -1, -1))
            self.B = self.B.expand(batch_size + (num_components, -1, -1))

    def compute_jacobian(self, sample_batch_dim, components, x, u):
        r"""
        TODO: Boris: Add docstring
        :param x:
        :param u:
        :return:
        """
        return self.A

    def compute_control_jacobian(self, sample_batch_dim, components, x, u):
        r"""
        TODO: Boris: Add docstring
        :param x:
        :param u:
        :return:
        """
        return self.B
