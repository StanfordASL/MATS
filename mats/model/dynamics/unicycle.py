import torch
from model.dynamics import Dynamic


class Unicycle(Dynamic):
    def init_constants(self, batch_size, num_components):
        pass

    def dynamic(self, x, u):
        r"""
        TODO: Boris: Add docstring
        :param x:
        :param u:
        :return:
        """
        x_p = x[..., 0]
        y_p = x[..., 1]
        phi = x[..., 2]
        v = x[..., 3]
        dphi = u[..., 0]
        a = u[..., 1]

        mask = torch.abs(dphi) <= 1e-2
        dphi = ~mask * dphi + (mask) * 1

        phi_p_omega_dt = phi + dphi * self.dt
        dsin_domega = (torch.sin(phi_p_omega_dt) - torch.sin(phi)) / dphi
        dcos_domega = (torch.cos(phi_p_omega_dt) - torch.cos(phi)) / dphi

        d1 = torch.stack([(x_p
                           + (a / dphi) * dcos_domega
                           + v * dsin_domega
                           + (a / dphi) * torch.sin(phi_p_omega_dt) * self.dt),
                          (y_p
                           - v * dcos_domega
                           + (a / dphi) * dsin_domega
                           - (a / dphi) * torch.cos(phi_p_omega_dt) * self.dt),
                          phi + dphi * self.dt,
                          v + a * self.dt], dim=-1)
        d2 = torch.stack([x_p + v * torch.cos(phi) * self.dt + (a / 2) * torch.cos(phi) * self.dt ** 2,
                          y_p + v * torch.sin(phi) * self.dt + (a / 2) * torch.sin(phi) * self.dt ** 2,
                          phi * torch.ones_like(a),
                          v + a * self.dt], dim=-1)
        return torch.where(~mask.unsqueeze(-1), d1, d2)

    def compute_control_jacobian(self, sample_batch_dim, components, x, u):
        r"""
        TODO: Boris: Add docstring
        :param x:
        :param u:
        :return:
        """
        F = torch.zeros(sample_batch_dim + (components, 4, 2),
                        device=self.device,
                        dtype=torch.float32)

        phi = x[..., 2]
        v = x[..., 3]
        dphi = u[0]
        a = u[1]

        mask = torch.abs(dphi) <= 1e-2
        dphi = ~mask * dphi + (mask) * 1

        phi_p_omega_dt = phi + dphi * self.dt
        dsin_domega = (torch.sin(phi_p_omega_dt) - torch.sin(phi)) / dphi
        dcos_domega = (torch.cos(phi_p_omega_dt) - torch.cos(phi)) / dphi

        F[..., 0, 0] = ((v / dphi) * torch.cos(phi_p_omega_dt) * self.dt
                        - (v / dphi) * dsin_domega
                        - (2 * a / dphi ** 2) * torch.sin(phi_p_omega_dt) * self.dt
                        - (2 * a / dphi ** 2) * dcos_domega
                        + (a / dphi) * torch.cos(phi_p_omega_dt) * self.dt ** 2)
        F[..., 0, 1] = (1 / dphi) * dcos_domega + (1 / dphi) * torch.sin(phi_p_omega_dt) * self.dt

        F[..., 1, 0] = ((v / dphi) * dcos_domega
                        - (2 * a / dphi ** 2) * dsin_domega
                        + (2 * a / dphi ** 2) * torch.cos(phi_p_omega_dt) * self.dt
                        + (v / dphi) * torch.sin(phi_p_omega_dt) * self.dt
                        + (a / dphi) * torch.sin(phi_p_omega_dt) * self.dt ** 2)
        F[..., 1, 1] = (1 / dphi) * dsin_domega - (1 / dphi) * torch.cos(phi_p_omega_dt) * self.dt

        F[..., 2, 0] = self.dt

        F[..., 3, 1] = self.dt

        F_sm = torch.zeros(sample_batch_dim + (components, 4, 2),
                           device=self.device,
                           dtype=torch.float32)

        F_sm[..., 0, 1] = (torch.cos(phi) * self.dt ** 2) / 2

        F_sm[..., 1, 1] = (torch.sin(phi) * self.dt ** 2) / 2

        F_sm[..., 3, 1] = self.dt

        return torch.where(~mask.unsqueeze(-1).unsqueeze(-1), F, F_sm)

    def compute_jacobian(self, sample_batch_dim, components, x, u):
        r"""
        TODO: Boris: Add docstring
        :param x:
        :param u:
        :return:
        """
        one = torch.tensor(1)
        F = torch.zeros(sample_batch_dim + (components, 4, 4),
                        device=self.device,
                        dtype=torch.float32)

        phi = x[..., 2]
        v = x[..., 3]
        dphi = u[0]
        a = u[1]

        mask = torch.abs(dphi) <= 1e-2
        dphi = ~mask * dphi + (mask) * 1

        phi_p_omega_dt = phi + dphi * self.dt
        dsin_domega = (torch.sin(phi_p_omega_dt) - torch.sin(phi)) / dphi
        dcos_domega = (torch.cos(phi_p_omega_dt) - torch.cos(phi)) / dphi

        F[..., 0, 0] = one
        F[..., 1, 1] = one
        F[..., 2, 2] = one
        F[..., 3, 3] = one

        F[..., 0, 2] = v * dcos_domega - (a / dphi) * dsin_domega + (a / dphi) * torch.cos(phi_p_omega_dt) * self.dt
        F[..., 0, 3] = dsin_domega

        F[..., 1, 2] = v * dsin_domega + (a / dphi) * dcos_domega + (a / dphi) * torch.sin(phi_p_omega_dt) * self.dt
        F[..., 1, 3] = -dcos_domega

        F_sm = torch.zeros(sample_batch_dim + (components, 4, 4),
                           device=self.device,
                           dtype=torch.float32)

        F_sm[..., 0, 0] = one
        F_sm[..., 1, 1] = one
        F_sm[..., 2, 2] = one
        F_sm[..., 3, 3] = one

        F_sm[..., 0, 2] = -v * torch.sin(phi) * self.dt - (a * torch.sin(phi) * self.dt ** 2) / 2
        F_sm[..., 0, 3] = torch.cos(phi) * self.dt

        F_sm[..., 1, 2] = v * torch.cos(phi) * self.dt + (a * torch.cos(phi) * self.dt ** 2) / 2
        F_sm[..., 1, 3] = torch.sin(phi) * self.dt

        return torch.where(~mask.unsqueeze(-1).unsqueeze(-1), F, F_sm)
