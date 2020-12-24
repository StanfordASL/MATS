import torch
import numpy as np
from model.mgcvae import ASubmodel, BQSubmodel
from model.dataset import get_timesteps_data
from model.model_utils import mutual_inf_mc, sigmoid_anneal
from model.components import GMM2D
import model.dynamics as dynamic_module
import torch.distributions as td
import model.anneal_scheduling as anneal_helper
from utils import calculate_A_slices, calculate_BQ_slices
from environment import derivative_of


class MATS(object):
    def __init__(self, model_registrar,
                 hyperparams, log_writer,
                 device):
        super(MATS, self).__init__()
        self.hyperparams = hyperparams
        self.log_writer = log_writer
        self.device = device
        self.curr_iter = 0

        self.model_registrar = model_registrar
        self.edge_models_dict = dict()
        self.node_models_dict = dict()
        self.dyn_classes = dict()
        self.nodes = set()

        self.env = None

        self.min_ht = self.hyperparams['minimum_history_length']
        self.max_ht = self.hyperparams['maximum_history_length']
        self.ph = self.hyperparams['prediction_horizon']

        self.state = self.hyperparams['state']
        self.state_length = dict()
        for state_type in self.state.keys():
            self.state_length[state_type] = int(
                np.sum([len(entity_dims) for entity_dims in self.state[state_type].values()])
            )

        self.pred_state = self.hyperparams['pred_state']
        self.pred_state_length = dict()
        for state_type in self.state.keys():
            self.pred_state_length[state_type] = int(
                np.sum([len(entity_dims) for entity_dims in self.pred_state[state_type].values()])
            )

    def set_environment(self, env):
        self.env = env

        self.edge_models_dict.clear()
        edge_types = env.get_edge_types()

        for edge_type in edge_types:
            orig_type, dest_type = edge_type
            a_submodel = ASubmodel(env,
                                   orig_type,
                                   dest_type,
                                   self.model_registrar,
                                   self.hyperparams,
                                   self.device,
                                   log_writer=self.log_writer)
            a_submodel.create_graphical_model()

            self.edge_models_dict[edge_type] = a_submodel

        for node_type in env.NodeType:
            bq_submodel = BQSubmodel(env,
                                     env.robot_type,
                                     node_type,
                                     self.model_registrar,
                                     self.hyperparams,
                                     self.device,
                                     log_writer=self.log_writer)
            bq_submodel.create_graphical_model()
            self.node_models_dict[node_type] = bq_submodel

            dynamic_class = getattr(dynamic_module, self.hyperparams['dynamic'][node_type]['name'])
            dyn_limits = self.hyperparams['dynamic'][node_type]['limits']
            self.dyn_classes[node_type.value] = dynamic_class(self.env.scenes[0].dt, dyn_limits, self.device,
                                                              self.model_registrar, node_type,
                                                              batch_size=(self.hyperparams['N'],),
                                                              num_components=self.hyperparams['K'])

    def set_curr_iter(self, curr_iter):
        self.curr_iter = curr_iter

        for A_submodel in self.edge_models_dict.values():
            A_submodel.set_curr_iter(curr_iter)

        for BQ_submodel in self.node_models_dict.values():
            BQ_submodel.set_curr_iter(curr_iter)

    def set_annealing_params(self):
        self.schedulers = list()
        self.annealed_vars = list()

        anneal_helper.create_new_scheduler(self,
                                           name='kl_weight',
                                           annealer=sigmoid_anneal,
                                           annealer_kws={
                                               'start': self.hyperparams['kl_weight_start'],
                                               'finish': self.hyperparams['kl_weight'],
                                               'center_step': self.hyperparams['kl_crossover'],
                                               'steps_lo_to_hi': self.hyperparams['kl_crossover'] / self.hyperparams[
                                                   'kl_sigmoid_divisor']
                                           })

        for A_submodel in self.edge_models_dict.values():
            A_submodel.set_annealing_params()

        for BQ_submodel in self.node_models_dict.values():
            BQ_submodel.set_annealing_params()

    def step_annealers(self):
        anneal_helper.step_annealers(self)

        for A_submodel in self.edge_models_dict.values():
            A_submodel.step_annealers()

        for BQ_submodel in self.node_models_dict.values():
            BQ_submodel.step_annealers()

        self.summarize_annealers()

    def summarize_annealers(self):
        anneal_helper.summarize_annealers(self, prefix='train')

    def extract_tensors_from_batch(self, batch, edge_tuple):
        (num_agents, state_lengths_in_order, agent_types_in_order,
         robot_index, node_type_count_dict, edge_type_count_dict,
         first_history_index_dict, state_lengths_dict,
         dest_indices_dict, orig_indices_dict,
         dest_inputs_t_dict, dest_labels_t_dict,
         dest_inputs_st_t_dict, orig_inputs_st_t_dict,
         dest_labels_st_t_dict,
         robot_traj_st_t_dict, map_tuples_dict,
         x_inits, prev_x_inits,
         robot_future_actions_st_t, y_targets) = batch

        # Getting the destination node's type since this is only used for BQ and those are
        # for when orig_node is the robot.
        node_type_count = node_type_count_dict[edge_tuple[1]]
        edge_type_count = edge_type_count_dict[edge_tuple]
        first_history_index = first_history_index_dict[edge_tuple]
        state_lengths = state_lengths_dict[edge_tuple]
        dest_indices = dest_indices_dict[edge_tuple]
        orig_indices = orig_indices_dict[edge_tuple]
        dest_inputs_t = dest_inputs_t_dict[edge_tuple]
        dest_labels_t = dest_labels_t_dict[edge_tuple]
        dest_inputs_st_t = dest_inputs_st_t_dict[edge_tuple]
        orig_inputs_st_t = orig_inputs_st_t_dict[edge_tuple]
        dest_labels_st_t = dest_labels_st_t_dict[edge_tuple]
        robot_traj_st_t = None
        if robot_traj_st_t_dict is not None:
            robot_traj_st_t = robot_traj_st_t_dict[edge_tuple]
        maps = None
        if map_tuples_dict is not None:
            maps = map_tuples_dict[edge_tuple]

        robot_index = robot_index.repeat(orig_indices.shape[1]).to(self.device)
        node_type_count = node_type_count.to(self.device)
        edge_type_count = edge_type_count.to(self.device)
        first_history_index = first_history_index.reshape((-1,)).to(self.device)
        state_lengths = state_lengths.to(self.device)
        dest_indices = dest_indices.reshape((-1,)).to(self.device)
        orig_indices = orig_indices.reshape((-1,)).to(self.device)
        dest_inputs_t = dest_inputs_t.reshape((-1,) + dest_inputs_t.shape[2:]).to(self.device)
        dest_labels_t = dest_labels_t.reshape((-1,) + dest_labels_t.shape[2:]).to(self.device)
        dest_inputs_st_t = dest_inputs_st_t.reshape((-1,) + dest_inputs_st_t.shape[2:]).to(self.device)
        orig_inputs_st_t = orig_inputs_st_t.reshape((-1,) + orig_inputs_st_t.shape[2:]).to(self.device)
        dest_labels_st_t = dest_labels_st_t.reshape((-1,) + dest_labels_st_t.shape[2:]).to(self.device)
        if robot_traj_st_t is not None:
            robot_traj_st_t = robot_traj_st_t.reshape((-1,) + robot_traj_st_t.shape[2:]).to(self.device)
        if maps is not None:
            maps = maps.reshape((-1,) + maps.shape[2:]).to(self.device)

        return (num_agents.to(self.device),
                state_lengths_in_order.to(self.device),
                robot_index,
                node_type_count,
                edge_type_count,
                first_history_index,
                state_lengths,
                dest_indices,
                orig_indices,
                dest_inputs_t,
                dest_labels_t,
                dest_inputs_st_t,
                orig_inputs_st_t,
                dest_labels_st_t,
                robot_traj_st_t,
                maps)

    def form_ABQ(self, A_outputs, BQ_outputs,
                 zero_R_rows=False):
        (ex_A_submatrices, _, _, _, num_agents, state_lengths_in_order, _) = A_outputs[0]
        orig_batch_size = num_agents.shape[0]
        ph = ex_A_submatrices.shape[1]
        num_samples = ex_A_submatrices.shape[2]
        num_components = ex_A_submatrices.shape[3]
        control_dim = BQ_outputs[0][0].shape[-1]

        (_, _, _, _, robot_index, _) = BQ_outputs[0]
        robot_index = robot_index[:orig_batch_size]

        As, Bs, Qs = [], [], []
        A_batch_idxs, BQ_batch_idxs = [], []

        summed_state_lengths_in_order = torch.zeros((state_lengths_in_order.shape[0],
                                                     state_lengths_in_order.shape[1] + 1),
                                                    dtype=state_lengths_in_order.dtype,
                                                    device=self.device)
        summed_state_lengths_in_order[:, 1:] = torch.cumsum(state_lengths_in_order, dim=1)

        num_edges_of_type = [0] * len(A_outputs)
        num_nodes_of_type = [0] * len(BQ_outputs)
        for batch_num in range(orig_batch_size):
            num_nodes = num_agents[batch_num]
            full_state_dim = state_lengths_in_order[batch_num, :num_nodes].sum()

            A = torch.full((ph, num_samples, num_components, full_state_dim, full_state_dim),
                           np.nan, dtype=torch.float, device=self.device)
            B = torch.full((ph, num_samples, num_components, full_state_dim, control_dim),
                           np.nan, dtype=torch.float, device=self.device)
            Q = torch.full((ph, num_samples, num_components, full_state_dim),
                           np.nan, dtype=torch.float, device=self.device)

            for idx, (A_submatrices, log_pis, orig_indices, dest_indices, _, _, edge_type_count) in enumerate(
                    A_outputs):
                curr_batch_idx = num_edges_of_type[idx]
                for i in range(edge_type_count[batch_num]):
                    rows, cols = calculate_A_slices(orig_indices[curr_batch_idx + i],
                                                    dest_indices[curr_batch_idx + i],
                                                    state_lengths_in_order[batch_num],
                                                    summed_state_lengths_in_order[batch_num])
                    if zero_R_rows and rows.start == 0:
                        A[..., rows, cols] = torch.zeros((ph, num_samples, num_components,
                                                          rows.stop - rows.start, cols.stop - cols.start),
                                                         dtype=torch.float, device=self.device)
                    else:
                        A[..., rows, cols] = A_submatrices[curr_batch_idx + i]

                A_batch_idxs.extend(
                    [batch_num] * edge_type_count[batch_num]
                )

                num_edges_of_type[idx] += edge_type_count[batch_num]

            # Note that node_type_count already excludes the robot,
            # so we don't need to subtract 1 from one of the counts.
            for idx, (B_submatrices, logQ_submatrices, log_pis, dest_indices, _, node_type_count) in enumerate(
                    BQ_outputs):
                curr_batch_idx = num_nodes_of_type[idx]
                for i in range(node_type_count[batch_num]):
                    rows = calculate_BQ_slices(dest_indices[curr_batch_idx + i],
                                               state_lengths_in_order[batch_num],
                                               summed_state_lengths_in_order[batch_num])

                    B[..., rows, :] = B_submatrices[curr_batch_idx + i]
                    Q[..., rows] = torch.exp(logQ_submatrices[curr_batch_idx + i])

                BQ_batch_idxs.extend(
                    [batch_num] * node_type_count[batch_num]
                )

                num_nodes_of_type[idx] += node_type_count[batch_num]

            # We have no uncertainty in the robot's next position.
            rows = calculate_BQ_slices(robot_index[batch_num],
                                       state_lengths_in_order[batch_num],
                                       summed_state_lengths_in_order[batch_num])
            Q[..., rows] = torch.zeros((ph, num_samples, num_components,
                                        self.pred_state_length[self.env.robot_type]),
                                       dtype=torch.float,
                                       device=self.device)

            As.append(A)
            Bs.append(B)
            Qs.append(Q)

        # Combining the batch assignments.
        A_batch_idxs = torch.tensor(A_batch_idxs, dtype=torch.int)
        BQ_batch_idxs = torch.tensor(BQ_batch_idxs, dtype=torch.int)

        return As, Bs, Qs, A_batch_idxs, BQ_batch_idxs

    def fill_AB(self, A, B, x, u_r, u_a,
                num_nodes, agent_types,
                state_lengths, summed_state_lengths,
                robot_index):
        sample_batch_dim = (A.shape[0],)
        components = A.shape[1]

        affine_term = torch.zeros_like(x)

        # Filling in the actual A and B dynamics matrices
        Brows = calculate_BQ_slices(robot_index, state_lengths, summed_state_lengths)
        B[..., Brows, :] = self.dyn_classes[self.env.robot_type.value].compute_control_jacobian(sample_batch_dim,
                                                                                                components,
                                                                                                x[..., Brows],
                                                                                                u_r)

        for agent_num in range(num_nodes):
            agent_type = agent_types[agent_num].item()
            Aidxs = calculate_BQ_slices(agent_num, state_lengths, summed_state_lengths)

            A[..., Aidxs, Aidxs] = self.dyn_classes[agent_type].compute_jacobian(sample_batch_dim,
                                                                                 components,
                                                                                 x[..., Aidxs],
                                                                                 u_r)

            if hasattr(self.env.NodeType, 'VEHICLE') and agent_type == self.env.NodeType.VEHICLE.value:
                # c = g(x_a, u_a) - Ax_a - Bu_a
                affine_term[..., Aidxs] = self.dyn_classes[agent_type].dynamic(x[..., Aidxs],
                                                                               u_a[...,
                                                                               agent_num * 2: (agent_num + 1) * 2])
                affine_term[..., Aidxs] -= (A[..., Aidxs, Aidxs] @ x[..., Aidxs].unsqueeze(-1)).squeeze(-1)
                affine_term[..., Aidxs] -= (
                            B[..., Aidxs, :] @ u_a[..., agent_num * 2: (agent_num + 1) * 2].unsqueeze(-1)).squeeze(-1)

        return affine_term

    def compute_agent_controls(self, x_prev, x_curr, u_a, agent_types_in_order, state_lengths, summed_state_lengths):
        for i, agent_type in enumerate(agent_types_in_order):
            if not hasattr(self.env.NodeType, 'VEHICLE') or agent_type != self.env.NodeType.VEHICLE.value:
                continue

            Aidxs = calculate_BQ_slices(i, state_lengths, summed_state_lengths)

            x_prev_a = x_prev[..., Aidxs]
            x_curr_a = x_curr[..., Aidxs]

            for sample in range(u_a.shape[0]):
                for num_components in range(u_a.shape[1]):
                    dphi = derivative_of(np.array([x_prev_a[sample, num_components, 2],
                                                   x_curr_a[sample, num_components, 2]]),
                                         dt=self.dyn_classes[agent_type.item()].dt,
                                         radian=True)[-1]
                    accel = derivative_of(np.array([x_prev_a[sample, num_components, 3],
                                                    x_curr_a[sample, num_components, 3]]),
                                          dt=self.dyn_classes[agent_type.item()].dt)[-1]

                    u_a[sample, num_components, i * 2] = dphi
                    u_a[sample, num_components, i * 2 + 1] = accel

    def integrate_LTV_systems(self, As, Bs, Qs, num_nodes, x_inits, prev_x_inits,
                              robot_future_actions_st_t, pis,
                              agent_types_in_order, state_lengths_in_order,
                              summed_state_lengths_in_order, robot_index,
                              return_mats=False, include_B=True):
        batch_size = len(As)
        timesteps = As[0].shape[0]

        ret_gmms = list()
        batched_non_rob_rows = list()
        if return_mats:
            filled_As, filled_Bs, affine_terms = list(), list(), list()

        for batch_num in range(batch_size):
            num_agents = num_nodes[batch_num]

            full_state_dim = As[batch_num].shape[-1]
            num_samples = As[batch_num].shape[1]
            num_components = As[batch_num].shape[2]

            if return_mats:
                filled_As.append(torch.empty_like(As[batch_num]))
                filled_Bs.append(torch.empty_like(Bs[batch_num]))
                affine_terms.append(torch.empty((timesteps, num_samples, num_components, full_state_dim),
                                                dtype=torch.float, device=self.device))

            means = torch.full((timesteps, num_samples, num_components, full_state_dim), np.nan,
                               dtype=torch.float, device=self.device)

            x_init = x_inits[batch_num][:full_state_dim].expand(As[batch_num].shape[1:3] + (-1,))
            prev_x_init = prev_x_inits[batch_num][:full_state_dim].expand(As[batch_num].shape[1:3] + (-1,))
            for timestep in range(timesteps):
                # Cloning is needed to make backpropagation work through this forloop.
                A = As[batch_num][timestep].clone()
                B = Bs[batch_num][timestep].clone()

                x = x_init if timestep == 0 else means[timestep - 1].clone()
                u_r = robot_future_actions_st_t[batch_num][timestep]
                u_a = np.zeros((num_samples, num_components, num_agents * 2))

                if timestep == 0:
                    self.compute_agent_controls(prev_x_init.detach().cpu().numpy(),
                                                x_init.detach().cpu().numpy(),
                                                u_a,
                                                agent_types_in_order[batch_num, :num_agents],
                                                state_lengths_in_order[batch_num],
                                                summed_state_lengths_in_order[batch_num])
                elif timestep == 1:
                    self.compute_agent_controls(x_init.detach().cpu().numpy(),
                                                means[timestep - 1].detach().cpu().numpy(),
                                                u_a,
                                                agent_types_in_order[batch_num, :num_agents],
                                                state_lengths_in_order[batch_num],
                                                summed_state_lengths_in_order[batch_num])
                else:
                    self.compute_agent_controls(means[timestep - 2].detach().cpu().numpy(),
                                                means[timestep - 1].detach().cpu().numpy(),
                                                u_a,
                                                agent_types_in_order[batch_num, :num_agents],
                                                state_lengths_in_order[batch_num],
                                                summed_state_lengths_in_order[batch_num])

                affine_term = self.fill_AB(A, B, x, u_r, torch.tensor(u_a, dtype=torch.float, device=self.device),
                                           num_agents,
                                           agent_types_in_order[batch_num],
                                           state_lengths_in_order[batch_num],
                                           summed_state_lengths_in_order[batch_num],
                                           robot_index[batch_num])

                means[timestep] = (A @ x.unsqueeze(-1)).squeeze(-1) + affine_term
                if include_B:
                    means[timestep] += B @ u_r

                if return_mats:
                    filled_As[-1][timestep] = A
                    filled_Bs[-1][timestep] = B
                    affine_terms[-1][timestep] = affine_term

            # For the GMMs we ignore the robot's means/covs since there's nothing we
            # can really do about them from a prediction perspective.
            rob_rows = calculate_BQ_slices(robot_index[batch_num],
                                           state_lengths_in_order[batch_num],
                                           summed_state_lengths_in_order[batch_num])
            non_rob_rows = torch.ones((means.shape[-1],), dtype=torch.bool)
            non_rob_rows[rob_rows] = False

            non_rob_means = means[..., non_rob_rows]
            cov_Ls = torch.diag_embed(Qs[batch_num][..., non_rob_rows])

            # Construct Gaussian Mixture Model in n-dimensions (the full state dimension) consisting of num_components
            # multivariate normal distributions weighted by pis.
            mix = td.Categorical(probs=pis[batch_num])
            comp = td.MultivariateNormal(loc=non_rob_means,
                                         scale_tril=cov_Ls)
            gmm = GMM2D(mix, comp)

            ret_gmms.append(gmm)
            batched_non_rob_rows.append(non_rob_rows)

        if return_mats:
            return ret_gmms, batched_non_rob_rows, filled_As, filled_Bs, affine_terms
        else:
            return ret_gmms, batched_non_rob_rows

    def aggregate_log_pis(self, combinedA_log_pis, combinedBQ_log_pis):
        pis = list()
        for batch_num in range(len(combinedA_log_pis)):
            A_log_pis = torch.sum(combinedA_log_pis[batch_num], dim=0)
            BQ_log_pis = torch.sum(combinedBQ_log_pis[batch_num], dim=0)
            log_pis = A_log_pis + BQ_log_pis
            pis.append(torch.softmax(log_pis, dim=-1))

        return torch.stack(pis, dim=0)

    def aggregate_values(self, combinedA_vals, combinedBQ_vals):
        values = list()
        for batch_num in range(len(combinedA_vals)):
            A_vals = torch.mean(combinedA_vals[batch_num], dim=0)
            BQ_vals = torch.mean(combinedBQ_vals[batch_num], dim=0)
            vals = (A_vals + BQ_vals) / 2
            values.append(torch.mean(vals, dim=-1))

        return torch.stack(values, dim=0)

    def train_loss(self, batch, include_B=True, reg_B=False, zero_R_rows=False):
        subA_p_log_pis = list()
        subBQ_p_log_pis = list()
        subA_q_log_pis = list()
        subBQ_q_log_pis = list()
        subA_kls = list()
        subBQ_kls = list()

        # Run forward pass
        A_outputs = list()
        BQ_outputs = list()
        for (orig_type, dest_type), A_submodel in self.edge_models_dict.items():
            edge_value_tuple = (orig_type.value, dest_type.value)

            (num_agents, state_lengths_in_order, robot_index,
             node_type_count, edge_type_count,
             first_history_index, state_lengths,
             dest_indices, orig_indices,
             dest_inputs_t, dest_labels_t,
             dest_inputs_st_t, orig_inputs_st_t,
             dest_labels_st_t, robot_traj_st_t,
             maps) = self.extract_tensors_from_batch(batch, edge_value_tuple)

            if torch.sum(edge_type_count) == 0:
                continue

            (kl,
             p_log_pis, q_log_pis,
             A_submatrices) = A_submodel.train_forward(dest_inputs=dest_inputs_t,
                                                       dest_inputs_st=dest_inputs_st_t,
                                                       first_history_indices=first_history_index,
                                                       dest_labels=dest_labels_t,
                                                       dest_labels_st=dest_labels_st_t,
                                                       orig_inputs=orig_inputs_st_t,
                                                       robot=robot_traj_st_t,
                                                       map=maps,
                                                       prediction_horizon=self.ph)

            selector = first_history_index > -1
            A_outputs.append((A_submatrices, q_log_pis, orig_indices[selector], dest_indices[selector],
                              num_agents, state_lengths_in_order, edge_type_count))

            subA_p_log_pis.append(p_log_pis)
            subA_q_log_pis.append(q_log_pis)
            subA_kls.append(kl)

            if edge_value_tuple[0] == self.env.robot_type.value:
                BQ_submodel = self.node_models_dict[dest_type]

                idxs = (orig_indices == robot_index)

                (kl,
                 p_log_pis, q_log_pis,
                 B_submatrices,
                 logQ_submatrices) = BQ_submodel.train_forward(dest_inputs=dest_inputs_t[idxs],
                                                               dest_inputs_st=dest_inputs_st_t[idxs],
                                                               first_history_indices=first_history_index[
                                                                   idxs],
                                                               dest_labels=dest_labels_t[idxs],
                                                               dest_labels_st=dest_labels_st_t[idxs],
                                                               orig_inputs=orig_inputs_st_t[idxs],
                                                               robot=robot_traj_st_t[idxs],
                                                               map=maps[idxs] if maps is not None else maps,
                                                               prediction_horizon=self.ph)

                BQ_outputs.append((B_submatrices, logQ_submatrices, q_log_pis,
                                   dest_indices[idxs], robot_index[idxs], node_type_count))

                subBQ_p_log_pis.append(p_log_pis)
                subBQ_q_log_pis.append(q_log_pis)
                subBQ_kls.append(kl)

        A_p_log_pis = torch.cat(subA_p_log_pis, dim=0)
        BQ_p_log_pis = torch.cat(subBQ_p_log_pis, dim=0)
        A_q_log_pis = torch.cat(subA_q_log_pis, dim=0)
        BQ_q_log_pis = torch.cat(subBQ_q_log_pis, dim=0)
        A_kls = torch.cat(subA_kls, dim=0)
        BQ_kls = torch.cat(subBQ_kls, dim=0)

        As, Bs, Qs, A_batch_idxs, BQ_batch_idxs = self.form_ABQ(A_outputs, BQ_outputs,
                                                                zero_R_rows=zero_R_rows)

        batch_size = len(As)
        combinedA_p_log_pis = list()
        combinedBQ_p_log_pis = list()
        combinedA_q_log_pis = list()
        combinedBQ_q_log_pis = list()
        combinedA_kls = list()
        combinedBQ_kls = list()
        for batch_num in range(batch_size):
            Aselector = A_batch_idxs == batch_num
            BQselector = BQ_batch_idxs == batch_num

            combinedA_p_log_pis.append(A_p_log_pis[Aselector])
            combinedBQ_p_log_pis.append(BQ_p_log_pis[BQselector])
            combinedA_q_log_pis.append(A_q_log_pis[Aselector])
            combinedBQ_q_log_pis.append(BQ_q_log_pis[BQselector])
            combinedA_kls.append(A_kls[Aselector])
            combinedBQ_kls.append(BQ_kls[BQselector])

        p_pis = self.aggregate_log_pis(combinedA_p_log_pis, combinedBQ_p_log_pis)
        q_pis = self.aggregate_log_pis(combinedA_q_log_pis, combinedBQ_q_log_pis)
        kls = self.aggregate_values(combinedA_kls, combinedBQ_kls)

        state_lengths_in_order, agent_types_in_order, robot_index = batch[1:4]
        x_inits, prev_x_inits, robot_future_actions_st_t, y_targets = batch[-4:]

        summed_state_lengths_in_order = torch.zeros((state_lengths_in_order.shape[0],
                                                     state_lengths_in_order.shape[1] + 1),
                                                    dtype=state_lengths_in_order.dtype,
                                                    device=self.device)
        summed_state_lengths_in_order[:, 1:] = torch.cumsum(state_lengths_in_order, dim=1)

        state_lengths_in_order = state_lengths_in_order.to(self.device)
        summed_state_lengths_in_order = summed_state_lengths_in_order.to(self.device)
        agent_types_in_order = agent_types_in_order.to(self.device)
        robot_index = robot_index.to(self.device)
        x_inits = x_inits.to(self.device)
        prev_x_inits = prev_x_inits.to(self.device)
        robot_future_actions_st_t = robot_future_actions_st_t.to(self.device)
        y_targets = y_targets.to(self.device)

        (pred_dists, non_rob_rows,
         filled_As, filled_Bs, affine_terms) = self.integrate_LTV_systems(As=As,
                                                                          Bs=Bs,
                                                                          Qs=Qs,
                                                                          num_nodes=batch[0],
                                                                          x_inits=x_inits,
                                                                          prev_x_inits=prev_x_inits,
                                                                          robot_future_actions_st_t=robot_future_actions_st_t,
                                                                          pis=q_pis,
                                                                          agent_types_in_order=agent_types_in_order,
                                                                          state_lengths_in_order=state_lengths_in_order,
                                                                          summed_state_lengths_in_order=summed_state_lengths_in_order,
                                                                          robot_index=robot_index,
                                                                          return_mats=True,
                                                                          include_B=include_B)

        if reg_B:
            B_regularization = 0.0

        if self.hyperparams['reg_A_slew']:
            A_regularization = 0.0

        # Distributions are in the original scale of the data.
        log_p_yt_xz = list()
        for batch_num, pred_dist in enumerate(pred_dists):
            y_target = y_targets[batch_num, ..., :state_lengths_in_order[batch_num, :num_agents[batch_num]].sum()]
            log_p_yt_xz.append(torch.clamp(pred_dist.log_prob(y_target[..., non_rob_rows[batch_num]].unsqueeze(1)),
                                           # min=self.hyperparams['log_p_yt_xz_min'],
                                           max=self.hyperparams['log_p_yt_xz_max']))

            if reg_B:
                B_regularization += torch.sum(torch.abs(filled_Bs[batch_num]))

            if self.hyperparams['reg_A_slew']:
                A_regularization += torch.sum(torch.abs(filled_As[batch_num][1:] - filled_As[batch_num][:-1]))

        log_p_yt_xz = torch.stack(log_p_yt_xz, dim=0)
        log_p_y_xz = torch.sum(log_p_yt_xz, dim=1)
        log_likelihood = torch.mean(log_p_y_xz)

        # kls is of shape [bs]
        kl_minibatch = torch.mean(kls, dim=0, keepdim=True)

        if self.log_writer is not None:
            self.log_writer.add_scalar('train/true_kl', torch.sum(kl_minibatch), self.curr_iter)

        if self.hyperparams['kl_min'] > 0:
            kl_lower_bounded = torch.clamp(kl_minibatch, min=self.hyperparams['kl_min'])
            kl = torch.sum(kl_lower_bounded)
        else:
            kl = torch.sum(kl_minibatch)

        # The timestep doesn't matter as the pis are the same across time.
        mutual_inf_p = mutual_inf_mc(td.Categorical(probs=p_pis[:, 0]))
        mutual_inf_q = mutual_inf_mc(td.Categorical(probs=q_pis[:, 0]))
        if self.log_writer is not None:
            self.log_writer.add_scalar('train/mutual_information_p',
                                       mutual_inf_p,
                                       self.curr_iter)
            self.log_writer.add_scalar('train/mutual_information_q',
                                       mutual_inf_q,
                                       self.curr_iter)

        ELBO = log_likelihood - self.kl_weight * kl + 1. * mutual_inf_p
        loss = -ELBO

        if reg_B:
            loss += self.hyperparams['reg_B_weight'] * B_regularization

        if self.hyperparams['reg_A_slew']:
            loss += self.hyperparams['reg_A_slew_weight'] * A_regularization

        return loss

    def eval_loss(self, batch, include_B, zero_R_rows):
        num_agents, y_targets = batch[0], batch[-1]
        (pred_dists, non_rob_rows,
         As, Bs, Qs, affine_terms,
         state_lengths_in_order,
         summed_state_lengths_in_order) = self.predict_forward(batch,
                                                               ph=y_targets.shape[1],
                                                               num_samples=1,
                                                               z_mode=False,
                                                               gmm_mode=False,
                                                               full_dist=True,
                                                               all_z_sep=False,
                                                               include_B=include_B,
                                                               zero_R_rows=zero_R_rows)

        # Distributions are in the original scale of the data.
        log_p_yt_xz = list()
        for batch_num, pred_dist in enumerate(pred_dists):
            y_target = y_targets[batch_num, ..., :state_lengths_in_order[batch_num, :num_agents[batch_num]].sum()]
            log_p_yt_xz.append(torch.clamp(pred_dist.log_prob(y_target[..., non_rob_rows[batch_num]].unsqueeze(1)),
                                           max=self.hyperparams['log_p_yt_xz_max']))

        log_p_yt_xz = torch.stack(log_p_yt_xz, dim=0)
        log_p_y_xz = torch.sum(log_p_yt_xz, dim=1)
        log_likelihood = torch.mean(log_p_y_xz)
        return -log_likelihood

    def convert_to_node_dict(self, scene, t,
                             predictions: GMM2D,
                             state_lengths_in_order,
                             summed_state_lengths_in_order):
        nodes = scene.present_nodes(np.array([t]),
                                    min_history_timesteps=1,
                                    min_future_timesteps=self.hyperparams['prediction_horizon'],
                                    return_robot=False)[t]
        pred_dict = dict()
        mixture_dist = td.Categorical(probs=predictions.mixture_distribution.probs.cpu())
        for idx, node in enumerate(sorted(nodes, key=lambda node: int(node.id))):
            rows = calculate_BQ_slices(idx, state_lengths_in_order, summed_state_lengths_in_order)
            component_dist = td.MultivariateNormal(loc=predictions.component_distribution.mean[..., rows].cpu(),
                                                   scale_tril=predictions.component_distribution.scale_tril[...,
                                                                                                            rows,
                                                                                                            rows].cpu())
            # Recall that the whole scene has the same z distribution.
            pred_dict[node] = GMM2D(mixture_distribution=mixture_dist,
                                    component_distribution=component_dist)
        return pred_dict

    def predict_forward(self,
                        batch,
                        ph,
                        num_samples=1,
                        z_mode=False,
                        gmm_mode=False,
                        full_dist=True,
                        all_z_sep=False,
                        include_B=True,
                        zero_R_rows=False):

        # Run forward pass
        A_outputs = list()
        BQ_outputs = list()
        subA_p_log_pis = list()
        subBQ_p_log_pis = list()
        for (orig_type, dest_type), A_submodel in self.edge_models_dict.items():
            edge_value_tuple = (orig_type.value, dest_type.value)

            (num_agents, state_lengths_in_order, robot_index,
             node_type_count, edge_type_count,
             first_history_index, state_lengths,
             dest_indices, orig_indices,
             dest_inputs_t, dest_labels_t,
             dest_inputs_st_t, orig_inputs_st_t,
             dest_labels_st_t, robot_traj_st_t,
             maps) = self.extract_tensors_from_batch(batch, edge_value_tuple)

            if torch.sum(edge_type_count) == 0:
                continue

            (p_log_pis,
             A_submatrices) = A_submodel.predict(dest_inputs=dest_inputs_t,
                                                 dest_inputs_st=dest_inputs_st_t,
                                                 first_history_indices=first_history_index,
                                                 dest_labels=dest_labels_t,
                                                 dest_labels_st=dest_labels_st_t,
                                                 orig_inputs=orig_inputs_st_t,
                                                 robot=robot_traj_st_t,
                                                 map=maps,
                                                 prediction_horizon=ph,
                                                 num_samples=num_samples,
                                                 z_mode=z_mode,
                                                 gmm_mode=gmm_mode,
                                                 full_dist=full_dist,
                                                 all_z_sep=all_z_sep)

            selector = first_history_index > -1
            A_outputs.append((A_submatrices, None, orig_indices[selector], dest_indices[selector],
                              num_agents, state_lengths_in_order, edge_type_count))

            subA_p_log_pis.append(p_log_pis)

            if edge_value_tuple[0] == self.env.robot_type.value:
                BQ_submodel = self.node_models_dict[dest_type]

                idxs = (orig_indices == robot_index)

                (p_log_pis,
                 B_submatrices,
                 logQ_submatrices) = BQ_submodel.predict(dest_inputs=dest_inputs_t[idxs],
                                                         dest_inputs_st=dest_inputs_st_t[idxs],
                                                         first_history_indices=first_history_index[
                                                             idxs],
                                                         dest_labels=dest_labels_t[idxs],
                                                         dest_labels_st=dest_labels_st_t[idxs],
                                                         orig_inputs=orig_inputs_st_t[idxs],
                                                         robot=robot_traj_st_t[idxs],
                                                         map=maps[idxs] if maps is not None else maps,
                                                         prediction_horizon=ph,
                                                         num_samples=num_samples,
                                                         z_mode=z_mode,
                                                         gmm_mode=gmm_mode,
                                                         full_dist=full_dist,
                                                         all_z_sep=all_z_sep)

                BQ_outputs.append((B_submatrices, logQ_submatrices, None,
                                   dest_indices[idxs], robot_index, node_type_count))

                subBQ_p_log_pis.append(p_log_pis)

        A_p_log_pis = torch.cat(subA_p_log_pis, dim=0)
        BQ_p_log_pis = torch.cat(subBQ_p_log_pis, dim=0)

        As, Bs, Qs, A_batch_idxs, BQ_batch_idxs = self.form_ABQ(A_outputs, BQ_outputs,
                                                                zero_R_rows=zero_R_rows)

        batch_size = len(As)
        combinedA_p_log_pis = list()
        combinedBQ_p_log_pis = list()
        for batch_num in range(batch_size):
            Aselector = A_batch_idxs == batch_num
            BQselector = BQ_batch_idxs == batch_num

            combinedA_p_log_pis.append(A_p_log_pis[Aselector])
            combinedBQ_p_log_pis.append(BQ_p_log_pis[BQselector])

        p_pis = self.aggregate_log_pis(combinedA_p_log_pis, combinedBQ_p_log_pis)

        state_lengths_in_order, agent_types_in_order, robot_index = batch[1:4]
        x_inits, prev_x_inits, robot_future_actions_st_t = batch[-4:-1]

        summed_state_lengths_in_order = torch.zeros((state_lengths_in_order.shape[0],
                                                     state_lengths_in_order.shape[1] + 1),
                                                    dtype=state_lengths_in_order.dtype,
                                                    device=self.device)
        summed_state_lengths_in_order[:, 1:] = torch.cumsum(state_lengths_in_order, dim=1)

        state_lengths_in_order = state_lengths_in_order.to(self.device)
        summed_state_lengths_in_order = summed_state_lengths_in_order.to(self.device)
        agent_types_in_order = agent_types_in_order.to(self.device)
        robot_index = robot_index.to(self.device)
        x_inits = x_inits.to(self.device)
        prev_x_inits = prev_x_inits.to(self.device)
        robot_future_actions_st_t = robot_future_actions_st_t.to(self.device)

        (pred_dists, non_rob_rows,
         filled_As, filled_Bs, affine_terms) = self.integrate_LTV_systems(As=As, Bs=Bs, Qs=Qs, num_nodes=batch[0],
                                                                          x_inits=x_inits, prev_x_inits=prev_x_inits,
                                                                          robot_future_actions_st_t=robot_future_actions_st_t,
                                                                          pis=p_pis,
                                                                          agent_types_in_order=agent_types_in_order,
                                                                          state_lengths_in_order=state_lengths_in_order,
                                                                          summed_state_lengths_in_order=summed_state_lengths_in_order,
                                                                          robot_index=robot_index, return_mats=True,
                                                                          include_B=include_B)

        return (pred_dists, non_rob_rows,
                filled_As, filled_Bs, Qs, affine_terms,
                state_lengths_in_order, summed_state_lengths_in_order)

    def predict(self,
                scene,
                timesteps,
                ph,
                num_samples=1,
                min_future_timesteps=0,
                min_history_timesteps=1,
                z_mode=False,
                gmm_mode=False,
                full_dist=True,
                all_z_sep=False,
                include_B=True,
                zero_R_rows=False):

        # Get Input data for node type and given timesteps
        batch, out_timesteps = get_timesteps_data(env=self.env, scene=scene, t=timesteps, state=self.state,
                                                  pred_state=self.pred_state,
                                                  min_ht=min_history_timesteps, max_ht=self.max_ht,
                                                  min_ft=min_future_timesteps, max_ft=min_future_timesteps,
                                                  hyperparams=self.hyperparams)

        (pred_dists, non_rob_rows,
         As, Bs, Qs, affine_terms,
         state_lengths_in_order,
         summed_state_lengths_in_order) = self.predict_forward(batch,
                                                               ph,
                                                               num_samples=num_samples,
                                                               z_mode=z_mode,
                                                               gmm_mode=gmm_mode,
                                                               full_dist=full_dist,
                                                               all_z_sep=all_z_sep,
                                                               include_B=include_B,
                                                               zero_R_rows=zero_R_rows)

        timestep_dict = dict()
        for idx, timestep in enumerate(out_timesteps):
            timestep_dict[timestep] = self.convert_to_node_dict(scene, timestep,
                                                                pred_dists[idx],
                                                                state_lengths_in_order[idx],
                                                                summed_state_lengths_in_order[idx])

        return timestep_dict, non_rob_rows, As, Bs, Qs, affine_terms, state_lengths_in_order.cpu()
