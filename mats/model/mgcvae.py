import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.components import *
from model.model_utils import ModeKeys, run_lstm_on_variable_length_seqs, unpack_RNN_state, exp_anneal, sigmoid_anneal
from environment.scene_graph import DirectedEdge
import model.anneal_scheduling as anneal_helper


class ASubmodel(object):
    def __init__(self,
                 env,
                 orig_type,
                 dest_type,
                 model_registrar,
                 hyperparams,
                 device,
                 log_writer=None):
        self.hyperparams = hyperparams
        self.env = env
        self.orig_type = orig_type
        self.dest_type = dest_type
        self.edge_type_str = DirectedEdge.get_str_from_types(orig_type, dest_type)
        self.model_registrar = model_registrar
        self.log_writer = log_writer
        self.device = device
        self.curr_iter = 0

        self.node_modules = dict()

        self.min_hl = self.hyperparams['minimum_history_length']
        self.max_hl = self.hyperparams['maximum_history_length']
        self.ph = self.hyperparams['prediction_horizon']
        self.state = self.hyperparams['state']
        self.pred_state = self.hyperparams['pred_state']

        self.orig_state_length = int(np.sum([len(entity_dims) for entity_dims in self.state[orig_type].values()]))
        self.dest_state_length = int(np.sum([len(entity_dims) for entity_dims in self.state[dest_type].values()]))

        self.orig_pred_state_length = int(
            np.sum([len(entity_dims) for entity_dims in self.pred_state[orig_type].values()]))
        self.dest_pred_state_length = int(
            np.sum([len(entity_dims) for entity_dims in self.pred_state[dest_type].values()]))

        if self.hyperparams['incl_robot_node']:
            self.robot_state_length = int(
                np.sum([len(entity_dims) for entity_dims in self.state[env.robot_type].values()]))
        self.pred_state_length = self.orig_pred_state_length * self.dest_pred_state_length

    def set_curr_iter(self, curr_iter):
        self.curr_iter = curr_iter

    def add_submodule(self, name, model_if_absent):
        self.node_modules[name] = self.model_registrar.get_model(name, model_if_absent)

    def clear_submodules(self):
        self.node_modules.clear()

    def create_submodules(self):
        ############################
        #   Node History Encoder   #
        ############################
        self.add_submodule(self.dest_type + '/node_history_encoder',
                           model_if_absent=nn.LSTM(input_size=self.dest_state_length,
                                                   hidden_size=self.hyperparams['enc_rnn_dim_history'],
                                                   batch_first=True))

        #####################
        #   Edge Encoders   #
        #####################
        if self.hyperparams['edge_encoding']:
            self.create_edge_model()

        ###########################
        #   Node Future Encoder   #
        ###########################
        # We'll create this here, but then later check if in training mode.
        # Based on that, we'll factor this into the computation graph (or not).
        self.add_submodule(self.dest_type + '/node_future_encoder',
                           model_if_absent=nn.LSTM(input_size=self.dest_state_length,
                                                   hidden_size=self.hyperparams['enc_rnn_dim_future'],
                                                   bidirectional=True,
                                                   batch_first=True))
        # These are related to how you initialize states for the node future encoder.
        self.add_submodule(self.dest_type + '/node_future_encoder/initial_h',
                           model_if_absent=nn.Linear(self.dest_state_length,
                                                     self.hyperparams['enc_rnn_dim_future']))
        self.add_submodule(self.dest_type + '/node_future_encoder/initial_c',
                           model_if_absent=nn.Linear(self.dest_state_length,
                                                     self.hyperparams['enc_rnn_dim_future']))

        ############################
        #   Robot Future Encoder   #
        ############################
        # We'll create this here, but then later check if we're next to the robot.
        # Based on that, we'll factor this into the computation graph (or not).
        # TODO: Check how to make the input_size a bit more general, right now everything works because
        #  all agent state lengths are the same.
        if self.hyperparams['incl_robot_node']:
            self.add_submodule(self.dest_type + '/robot_future_encoder',
                               model_if_absent=nn.LSTM(input_size=self.robot_state_length,
                                                       hidden_size=self.hyperparams['enc_rnn_dim_future'],
                                                       bidirectional=True,
                                                       batch_first=True))
            # These are related to how you initialize states for the robot future encoder.
            self.add_submodule(self.dest_type + '/robot_future_encoder/initial_h',
                               model_if_absent=nn.Linear(self.robot_state_length,
                                                         self.hyperparams['enc_rnn_dim_future']))
            self.add_submodule(self.dest_type + '/robot_future_encoder/initial_c',
                               model_if_absent=nn.Linear(self.robot_state_length,
                                                         self.hyperparams['enc_rnn_dim_future']))

        ###################
        #   Map Encoder   #
        ###################
        if self.hyperparams['use_map_encoding']:
            if self.dest_type in self.hyperparams['map_encoder']:
                me_params = self.hyperparams['map_encoder'][self.dest_type]
                self.add_submodule(self.dest_type + '/map_encoder',
                                   model_if_absent=CNNMapEncoder(me_params['map_channels'],
                                                                 me_params['hidden_channels'],
                                                                 me_params['output_size'],
                                                                 me_params['masks'],
                                                                 me_params['strides'],
                                                                 me_params['patch_size']))

        ################################
        #   Discrete Latent Variable   #
        ################################
        self.latent = DiscreteLatent(self.hyperparams, self.device)

        ######################################################################
        #   Various Fully-Connected Layers from Encoder to Latent Variable   #
        ######################################################################
        # Node History Encoder
        x_size = self.hyperparams['enc_rnn_dim_history']
        if self.hyperparams['edge_encoding']:
            #              Edge Encoder
            x_size += self.hyperparams['enc_rnn_dim_edge']
        if self.hyperparams['incl_robot_node']:
            #              Future Conditional Encoder
            x_size += 4 * self.hyperparams['enc_rnn_dim_future']
        if self.hyperparams['use_map_encoding'] and self.dest_type in self.hyperparams['map_encoder']:
            #              Map Encoder
            x_size += self.hyperparams['map_encoder'][self.dest_type]['output_size']

        z_size = self.hyperparams['N'] * self.hyperparams['K']

        if self.hyperparams['p_z_x_MLP_dims'] is not None:
            self.add_submodule(self.edge_type_str + '/p_z_x',
                               model_if_absent=nn.Linear(x_size, self.hyperparams['p_z_x_MLP_dims']))
            hx_size = self.hyperparams['p_z_x_MLP_dims']
        else:
            hx_size = x_size

        self.add_submodule(self.edge_type_str + '/hx_to_z',
                           model_if_absent=nn.Linear(hx_size, self.latent.z_dim))

        if self.hyperparams['q_z_xy_MLP_dims'] is not None:
            self.add_submodule(self.edge_type_str + '/q_z_xy',
                               #                                           Node Future Encoder
                               model_if_absent=nn.Linear(x_size + 4 * self.hyperparams['enc_rnn_dim_future'],
                                                         self.hyperparams['q_z_xy_MLP_dims']))
            hxy_size = self.hyperparams['q_z_xy_MLP_dims']
        else:
            #                           Node Future Encoder
            hxy_size = x_size + 4 * self.hyperparams['enc_rnn_dim_future']

        self.add_submodule(self.edge_type_str + '/hxy_to_z',
                           model_if_absent=nn.Linear(hxy_size, self.latent.z_dim))

        self.x_size = x_size
        self.z_size = z_size

    def create_edge_model(self):
        neighbor_state_length = int(
            np.sum([len(entity_dims) for entity_dims in self.state[self.edge_type_str.split('->')[1]].values()]))

        edge_encoder_input_size = self.orig_state_length + neighbor_state_length

        self.add_submodule(self.edge_type_str + '/edge_encoder',
                           model_if_absent=nn.LSTM(input_size=edge_encoder_input_size,
                                                   hidden_size=self.hyperparams['enc_rnn_dim_edge'],
                                                   batch_first=True))

    def create_decoder_submodules(self):
        if self.hyperparams['incl_robot_node']:
            decoder_input_dims = self.pred_state_length + self.robot_state_length + self.z_size + self.x_size
        else:
            decoder_input_dims = self.pred_state_length + self.z_size + self.x_size

        self.add_submodule(self.edge_type_str + '/decoder/A/initial_input',
                           model_if_absent=nn.Sequential(
                               nn.Linear(self.orig_state_length + self.dest_state_length, self.pred_state_length)))

        self.add_submodule(self.edge_type_str + '/decoder/A/rnn_cell',
                           model_if_absent=nn.GRUCell(decoder_input_dims, self.hyperparams['dec_rnn_dim']))
        self.add_submodule(self.edge_type_str + '/decoder/A/initial_h',
                           model_if_absent=nn.Linear(self.z_size + self.x_size, self.hyperparams['dec_rnn_dim']))

        self.add_submodule(self.edge_type_str + '/decoder/A/proj_to_submatrix',
                           model_if_absent=nn.Linear(self.hyperparams['dec_rnn_dim'],
                                                     self.pred_state_length))

    def create_graphical_model(self):
        """
        Creates or queries all trainable components.

        :return: None
        """
        self.clear_submodules()

        ##############################
        #   Everything but Decoder   #
        ##############################
        self.create_submodules()

        ###############
        #   Decoder   #
        ###############
        self.create_decoder_submodules()

        for name, module in self.node_modules.items():
            module.to(self.device)

    def set_annealing_params(self):
        self.schedulers = list()
        self.annealed_vars = list()

        anneal_helper.create_new_scheduler(self,
                                           name='latent.temp',
                                           annealer=exp_anneal,
                                           annealer_kws={
                                               'start': self.hyperparams['tau_init'],
                                               'finish': self.hyperparams['tau_final'],
                                               'rate': self.hyperparams['tau_decay_rate']
                                           })

        anneal_helper.create_new_scheduler(self,
                                           name='latent.z_logit_clip',
                                           annealer=sigmoid_anneal,
                                           annealer_kws={
                                               'start': self.hyperparams['z_logit_clip_start'],
                                               'finish': self.hyperparams['z_logit_clip_final'],
                                               'center_step': self.hyperparams['z_logit_clip_crossover'],
                                               'steps_lo_to_hi': self.hyperparams['z_logit_clip_crossover'] /
                                                                 self.hyperparams[
                                                                     'z_logit_clip_divisor']
                                           },
                                           creation_condition=self.hyperparams['use_z_logit_clipping'])

    def step_annealers(self):
        anneal_helper.step_annealers(self)
        self.summarize_annealers()

    def summarize_annealers(self):
        anneal_helper.summarize_annealers(self, prefix=self.edge_type_str)

    def obtain_encoded_tensors(self,
                               mode,
                               dest_inputs,
                               dest_inputs_st,
                               dest_labels,
                               dest_labels_st,
                               first_history_indices,
                               orig_history_st,
                               robot,
                               map) -> (torch.Tensor,
                                        torch.Tensor,
                                        torch.Tensor,
                                        torch.Tensor,
                                        torch.Tensor,
                                        torch.Tensor):
        """
        Encodes input and output tensors for node and robot.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param dest_inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param dest_inputs_st: Standardized input tensor.
        :param dest_labels: Label tensor including the label output for each agent over time [bs, t, pred_state].
        :param dest_labels_st: Standardized label tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param orig_history_st: Preprocessed dict (indexed by edge type) of list of neighbor states over time.
                            [[bs, t, neighbor state]]
        :param robot: Standardized robot state over time. [bs, t, robot_state]
        :param map: Tensor of Map information. [bs, channels, x, y]
        :return: tuple(x, x_nr_t, y_e, y_r, y, n_s_t0)
            WHERE
            - x: Encoded input / condition tensor to the CVAE x_e.
            - x_r_t: Robot state (if robot is in scene).
            - y_e: Encoded label / future of the node.
            - y_r: Encoded future of the robot.
            - y: Label / future of the node.
            - n_s_t0: Standardized current state of the node.
        """

        x, x_r_t, y_e, y_r, y = None, None, None, None, None
        selector = first_history_indices > -1

        #########################################
        # Provide basic information to encoders #
        #########################################
        fhi_selected = first_history_indices[selector]
        node_history = dest_inputs[selector]
        node_history_st = dest_inputs_st[selector]
        node_present_state_st = dest_inputs_st[selector, -1]
        other_history_st = orig_history_st[selector]

        n_s_t0 = torch.cat([node_present_state_st, other_history_st[:, -1]], dim=-1)

        if self.hyperparams['incl_robot_node']:
            x_r_t, y_r = robot[selector, ..., 0, :], robot[selector, ..., 1:, :]

        ##################
        # Encode History #
        ##################
        node_history_encoded = self.encode_node_history(mode,
                                                        node_history_st,
                                                        fhi_selected)

        ##################
        # Encode Present #
        ##################
        node_present = node_present_state_st  # [bs, state_dim]

        ##################
        # Encode Future #
        ##################
        if mode != ModeKeys.PREDICT:
            y = dest_labels_st[selector]

        #####################
        # Encode Node Edges #
        #####################
        if self.hyperparams['edge_encoding']:
            # Encode edges for given edge type
            total_edge_influence = self.encode_edge(mode,
                                                    node_history,
                                                    node_history_st,
                                                    self.edge_type_str,
                                                    other_history_st,
                                                    fhi_selected)

        ################
        # Map Encoding #
        ################
        if self.hyperparams['use_map_encoding'] and self.dest_type in self.hyperparams['map_encoder']:
            if self.log_writer and (self.curr_iter + 1) % 500 == 0:
                map_clone = map.clone()
                map_patch = self.hyperparams['map_encoder'][self.dest_type]['patch_size']
                map_clone[:, :, map_patch[1] - 5:map_patch[1] + 5, map_patch[0] - 5:map_patch[0] + 5] = 1.
                self.log_writer.add_images(f"{self.dest_type}/cropped_maps", map_clone,
                                           self.curr_iter, dataformats='NCWH')

            encoded_map = self.node_modules[self.dest_type + '/map_encoder'](map * 2. - 1., (mode == ModeKeys.TRAIN))
            do = self.hyperparams['map_encoder'][self.dest_type]['dropout']
            encoded_map = F.dropout(encoded_map, do, training=(mode == ModeKeys.TRAIN))

        ######################################
        # Concatenate Encoder Outputs into x #
        ######################################
        x_concat_list = list()

        # Every node has an edge-influence encoder (which could just be zero).
        if self.hyperparams['edge_encoding']:
            x_concat_list.append(total_edge_influence)  # [bs/nbs, 4*enc_rnn_dim]

        # Every node has a history encoder.
        x_concat_list.append(node_history_encoded)  # [bs/nbs, enc_rnn_dim_history]

        if self.hyperparams['incl_robot_node']:
            robot_future_encoder = self.encode_robot_future(mode, x_r_t, y_r)
            x_concat_list.append(robot_future_encoder)

        if self.hyperparams['use_map_encoding'] and self.dest_type in self.hyperparams['map_encoder']:
            if self.log_writer:
                self.log_writer.add_scalar(f"{self.dest_type}/encoded_map_max",
                                           torch.max(torch.abs(encoded_map)), self.curr_iter)
            x_concat_list.append(encoded_map)

        x = torch.cat(x_concat_list, dim=1)

        if mode == ModeKeys.TRAIN or mode == ModeKeys.EVAL:
            y_e = self.encode_node_future(mode, node_present, y)

        return x, x_r_t, y_e, y_r, y, n_s_t0

    def encode_node_history(self, mode, node_hist, first_history_indices):
        """
        Encodes the nodes history.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param node_hist: Historic and current state of the node. [bs, mhl, state]
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :return: Encoded node history tensor. [bs, enc_rnn_dim]
        """
        outputs, _ = run_lstm_on_variable_length_seqs(self.node_modules[self.dest_type + '/node_history_encoder'],
                                                      original_seqs=node_hist,
                                                      lower_indices=first_history_indices)

        outputs = F.dropout(outputs,
                            p=1. - self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                            training=(mode == ModeKeys.TRAIN))  # [bs, max_time, enc_rnn_dim]

        last_index_per_sequence = -(first_history_indices + 1)

        return outputs[torch.arange(first_history_indices.shape[0]), last_index_per_sequence]

    def encode_edge(self,
                    mode,
                    node_history,
                    node_history_st,
                    edge_type_str,
                    neighbor_history_st,
                    first_history_indices):

        combined_neighbors = neighbor_history_st  # [bs, max_ht, state_dim]

        joint_history = torch.cat([combined_neighbors, node_history_st], dim=-1)

        outputs, _ = run_lstm_on_variable_length_seqs(
            self.node_modules[edge_type_str + '/edge_encoder'],
            original_seqs=joint_history,
            lower_indices=first_history_indices
        )

        outputs = F.dropout(outputs,
                            p=1. - self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                            training=(mode == ModeKeys.TRAIN))  # [bs, max_time, enc_rnn_dim]

        last_index_per_sequence = -(first_history_indices + 1)
        ret = outputs[torch.arange(last_index_per_sequence.shape[0]), last_index_per_sequence]
        return ret

    def encode_node_future(self, mode, node_present, node_future) -> torch.Tensor:
        """
        Encodes the node future (during training) using a bi-directional LSTM

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param node_present: Current state of the node. [bs, state]
        :param node_future: Future states of the node. [bs, ph, state]
        :return: Encoded future.
        """
        initial_h_model = self.node_modules[self.dest_type + '/node_future_encoder/initial_h']
        initial_c_model = self.node_modules[self.dest_type + '/node_future_encoder/initial_c']

        # Here we're initializing the forward hidden states,
        # but zeroing the backward ones.
        initial_h = initial_h_model(node_present)
        initial_h = torch.stack([initial_h, torch.zeros_like(initial_h, device=self.device)], dim=0)

        initial_c = initial_c_model(node_present)
        initial_c = torch.stack([initial_c, torch.zeros_like(initial_c, device=self.device)], dim=0)

        initial_state = (initial_h, initial_c)

        _, state = self.node_modules[self.dest_type + '/node_future_encoder'](node_future, initial_state)
        state = unpack_RNN_state(state)
        state = F.dropout(state,
                          p=1. - self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                          training=(mode == ModeKeys.TRAIN))

        return state

    def encode_robot_future(self, mode, robot_present, robot_future) -> torch.Tensor:
        """
        Encodes the robot future (during training) using a bi-directional LSTM

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param robot_present: Current state of the robot. [bs, state]
        :param robot_future: Future states of the robot. [bs, ph, state]
        :return: Encoded future.
        """
        initial_h_model = self.node_modules[self.dest_type + '/robot_future_encoder/initial_h']
        initial_c_model = self.node_modules[self.dest_type + '/robot_future_encoder/initial_c']

        # Here we're initializing the forward hidden states,
        # but zeroing the backward ones.
        initial_h = initial_h_model(robot_present)
        initial_h = torch.stack([initial_h, torch.zeros_like(initial_h, device=self.device)], dim=0)

        initial_c = initial_c_model(robot_present)
        initial_c = torch.stack([initial_c, torch.zeros_like(initial_c, device=self.device)], dim=0)

        initial_state = (initial_h, initial_c)

        _, state = self.node_modules[self.dest_type + '/robot_future_encoder'](robot_future, initial_state)
        state = unpack_RNN_state(state)
        state = F.dropout(state,
                          p=1. - self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                          training=(mode == ModeKeys.TRAIN))

        return state

    def q_z_xy(self, mode, x, y_e) -> torch.Tensor:
        r"""
        .. math:: q_\phi(z \mid \mathbf{x}_i, \mathbf{y}_i)

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :param y_e: Encoded future tensor.
        :return: Latent distribution of the CVAE.
        """
        xy = torch.cat([x, y_e], dim=1)

        if self.hyperparams['q_z_xy_MLP_dims'] is not None:
            dense = self.node_modules[self.edge_type_str + '/q_z_xy']
            h = F.dropout(F.relu(dense(xy)),
                          p=1. - self.hyperparams['MLP_dropout_keep_prob'],
                          training=(mode == ModeKeys.TRAIN))

        else:
            h = xy

        to_latent = self.node_modules[self.edge_type_str + '/hxy_to_z']
        return self.latent.dist_from_h(to_latent(h), mode)

    def p_z_x(self, mode, x):
        r"""
        .. math:: p_\theta(z \mid \mathbf{x}_i)

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :return: Latent distribution of the CVAE.
        """
        if self.hyperparams['p_z_x_MLP_dims'] is not None:
            dense = self.node_modules[self.edge_type_str + '/p_z_x']
            h = F.dropout(F.relu(dense(x)),
                          p=1. - self.hyperparams['MLP_dropout_keep_prob'],
                          training=(mode == ModeKeys.TRAIN))

        else:
            h = x

        to_latent = self.node_modules[self.edge_type_str + '/hx_to_z']
        return self.latent.dist_from_h(to_latent(h), mode)

    def p_y_xz(self, mode, x, x_nr_t, y_r, n_s_t0, z_stacked, prediction_horizon,
               num_samples, num_components=1, gmm_mode=False):
        r"""
        .. math:: p_\psi(\mathbf{y}_i \mid \mathbf{x}_i, z)

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :param x_nr_t: Joint state of node and robot (if robot is in scene).
        :param y: Future tensor.
        :param y_r: Encoded future tensor.
        :param n_s_t0: Standardized current state of the node.
        :param z_stacked: Stacked latent state. [num_samples_z * num_samples_gmm, bs, latent_state]
        :param prediction_horizon: Number of prediction timesteps.
        :param num_samples: Number of samples from the latent space.
        :param num_components: Number of GMM components.
        :param gmm_mode: If True: The mode of the GMM is sampled.
        :return: GMM2D. If mode is Predict, also samples from the GMM.
        """
        ph = prediction_horizon

        z = torch.reshape(z_stacked, (-1, self.latent.z_dim))
        zx = torch.cat([z, x.repeat(num_samples * num_components, 1)], dim=1)

        cell = self.node_modules[self.edge_type_str + '/decoder/A/rnn_cell']
        initial_h_model = self.node_modules[self.edge_type_str + '/decoder/A/initial_h']

        initial_state = initial_h_model(zx)

        A_submatrices = []

        # log_pis is [bs, ph, num_samples, num_components]
        p_log_pis = self.latent.p_dist.logits.unsqueeze(1).repeat(num_samples, ph, 1, 1)
        q_log_pis = None
        if mode != ModeKeys.PREDICT:
            q_log_pis = self.latent.q_dist.logits.unsqueeze(1).repeat(num_samples, ph, 1, 1)

        # Infer initial action state for node from current state
        a_0 = self.node_modules[self.edge_type_str + '/decoder/A/initial_input'](n_s_t0)

        state = initial_state
        if self.hyperparams['incl_robot_node']:
            input_ = torch.cat([zx,
                                a_0.repeat(num_samples * num_components, 1),
                                x_nr_t.repeat(num_samples * num_components, 1)], dim=1)
        else:
            input_ = torch.cat([zx, a_0.repeat(num_samples * num_components, 1)], dim=1)

        for j in range(ph):
            h_state = cell(input_, state)
            A_submatrix_vector = self.node_modules[self.edge_type_str + '/decoder/A/proj_to_submatrix'](h_state)

            A_submatrices.append(A_submatrix_vector / self.hyperparams['init_scale'])

            if self.hyperparams['incl_robot_node']:
                dec_inputs = [zx, A_submatrix_vector, y_r[:, j].repeat(num_samples * num_components, 1)]
            else:
                dec_inputs = [zx, A_submatrix_vector]
            input_ = torch.cat(dec_inputs, dim=1)
            state = h_state

        # A_submatrices is [num_samples, num_components, bs, ph, dest_pred_state_length, orig_pred_state_length]
        A_submatrices = torch.stack(A_submatrices, dim=1)
        A_submatrices = torch.reshape(A_submatrices,
                                      [num_samples, num_components,
                                       -1, ph,
                                       self.dest_pred_state_length,
                                       self.orig_pred_state_length]).permute(2, 3, 0, 1, 4, 5)

        # log_pis is now [bs, ph, num_samples, num_components]
        # A_submatrices is now [bs, ph, num_samples, num_components, dest_pred_state_length, orig_pred_state_length]
        return p_log_pis, q_log_pis, A_submatrices

    def encoder(self, mode, x, y_e, num_samples=None):
        """
        Encoder of the CVAE.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :param y_e: Encoded future tensor.
        :param num_samples: Number of samples from the latent space during Prediction.
        :return: tuple(z, kl_obj)
            WHERE
            - z: Samples from the latent space.
            - kl_obj: KL Divergenze between q and p
        """
        if mode == ModeKeys.TRAIN:
            sample_ct = self.hyperparams['k']
        elif mode == ModeKeys.EVAL:
            sample_ct = self.hyperparams['k_eval']
        elif mode == ModeKeys.PREDICT:
            sample_ct = num_samples
            if num_samples is None:
                raise ValueError("num_samples cannot be None with mode == PREDICT.")

        self.latent.q_dist = self.q_z_xy(mode, x, y_e)
        self.latent.p_dist = self.p_z_x(mode, x)

        z = self.latent.sample_q(sample_ct, mode)

        if mode == ModeKeys.TRAIN:
            kl_obj = self.latent.kl_q_p(self.log_writer, '%s' % str(self.edge_type_str), self.curr_iter)
            # if self.log_writer is not None:
            #     self.log_writer.add_scalar('%s/%s' % (str(self.edge_type_str), 'kl'), kl_obj, self.curr_iter)
        else:
            kl_obj = None

        return z, kl_obj

    def decoder(self, mode, x, x_nr_t, y, y_r, n_s_t0, z, prediction_horizon, num_samples):
        """
        Decoder of the CVAE.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :param x: Input / Condition tensor.
        :param x_nr_t: Joint state of node and robot (if robot is in scene).
        :param y: Future tensor.
        :param y_r: Encoded future tensor.
        :param n_s_t0: Standardized current state of the node.
        :param z: Stacked latent state.
        :param prediction_horizon: Number of prediction timesteps.
        :param num_samples: Number of samples from the latent space.
        :return: Log probability of y over p.
        """
        num_components = self.hyperparams['N'] * self.hyperparams['K']
        p_log_pis, q_log_pis, A_submatrices = self.p_y_xz(mode, x, x_nr_t, y_r, n_s_t0, z,
                                                          prediction_horizon, num_samples,
                                                          num_components=num_components)
        return p_log_pis, q_log_pis, A_submatrices

    def train_forward(self,
                      dest_inputs,
                      dest_inputs_st,
                      first_history_indices,
                      dest_labels,
                      dest_labels_st,
                      orig_inputs,
                      robot,
                      map,
                      prediction_horizon) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Calculates the training loss for a batch.

        :param dest_inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param dest_inputs_st: Standardized input tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param dest_labels: Label tensor including the label output for each agent over time [bs, t, pred_state].
        :param dest_labels_st: Standardized label tensor.
        :param orig_inputs: Preprocessed dict (indexed by edge type) of list of neighbor states over time.
                            [[bs, t, neighbor state]]
        :param robot: Standardized robot state over time. [bs, t, robot_state]
        :param map: Tensor of Map information. [bs, channels, x, y]
        :param prediction_horizon: Number of prediction timesteps.
        :return: Scalar tensor -> nll loss
        """
        mode = ModeKeys.TRAIN

        x, x_nr_t, y_e, y_r, y, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                                     dest_inputs=dest_inputs,
                                                                     dest_inputs_st=dest_inputs_st,
                                                                     dest_labels=dest_labels,
                                                                     dest_labels_st=dest_labels_st,
                                                                     first_history_indices=first_history_indices,
                                                                     orig_history_st=orig_inputs,
                                                                     robot=robot,
                                                                     map=map)

        z, kl = self.encoder(mode, x, y_e)
        p_log_pis, q_log_pis, A_submatrices = self.decoder(mode, x, x_nr_t, y, y_r, n_s_t0, z,
                                                           prediction_horizon,
                                                           self.hyperparams['k'])

        if self.log_writer is not None:
            if self.hyperparams['log_histograms']:
                self.latent.summarize_for_tensorboard(self.log_writer, str(self.edge_type_str), self.curr_iter)

        return kl, p_log_pis, q_log_pis, A_submatrices

    def predict(self,
                dest_inputs,
                dest_inputs_st,
                first_history_indices,
                dest_labels,
                dest_labels_st,
                orig_inputs,
                robot,
                map,
                prediction_horizon,
                num_samples,
                z_mode=False,
                gmm_mode=False,
                full_dist=True,
                all_z_sep=False):
        """
        Predicts the future of a batch of nodes.

        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param neighbors: Preprocessed dict (indexed by edge type) of list of neighbor states over time.
                            [[bs, t, neighbor state]]
        :param neighbors_edge_value: Preprocessed edge values for all neighbor nodes [[N]]
        :param robot: Standardized robot state over time. [bs, t, robot_state]
        :param map: Tensor of Map information. [bs, channels, x, y]
        :param prediction_horizon: Number of prediction timesteps.
        :param num_samples: Number of samples from the latent space.
        :param z_mode: If True: Select the most likely latent state.
        :param gmm_mode: If True: The mode of the GMM is sampled.
        :param all_z_sep: Samples each latent mode individually without merging them into a GMM.
        :param full_dist: Samples all latent states and merges them into a GMM as output.
        :return:
        """
        mode = ModeKeys.PREDICT

        x, x_nr_t, _, y_r, _, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                                   dest_inputs=dest_inputs,
                                                                   dest_inputs_st=dest_inputs_st,
                                                                   dest_labels=dest_labels,
                                                                   dest_labels_st=dest_labels_st,
                                                                   first_history_indices=first_history_indices,
                                                                   orig_history_st=orig_inputs,
                                                                   robot=robot,
                                                                   map=map)

        self.latent.p_dist = self.p_z_x(mode, x)
        z, num_samples, num_components = self.latent.sample_p(num_samples,
                                                              mode,
                                                              most_likely_z=z_mode,
                                                              full_dist=full_dist,
                                                              all_z_sep=all_z_sep)

        p_log_pis, _, A_submatrices = self.p_y_xz(mode, x, x_nr_t, y_r, n_s_t0, z,
                                                  prediction_horizon,
                                                  num_samples,
                                                  num_components,
                                                  gmm_mode)

        return p_log_pis, A_submatrices


class BQSubmodel(ASubmodel):
    def __init__(self,
                 env,
                 orig_type,
                 dest_type,
                 model_registrar,
                 hyperparams,
                 device,
                 log_writer=None):
        super(BQSubmodel, self).__init__(env,
                                         orig_type,
                                         dest_type,
                                         model_registrar,
                                         hyperparams,
                                         device,
                                         log_writer)

        self.pred_state_length = self.dest_pred_state_length * 2 + self.dest_pred_state_length

    def create_decoder_submodules(self):
        if self.hyperparams['incl_robot_node']:
            decoder_input_dims = self.pred_state_length + self.robot_state_length + self.z_size + self.x_size
        else:
            decoder_input_dims = self.pred_state_length + self.z_size + self.x_size

        # Yes, the model is named "BQ" and these strings are "BC", it's because we referred to the uncertainty matrix
        # as C in early development and so all the weights were saved according to the "BC" naming scheme. Variables
        # and etc in the code should all be "BQ" now though!
        self.add_submodule(self.edge_type_str + '/decoder/BC/initial_input',
                           model_if_absent=nn.Sequential(
                               nn.Linear(self.orig_state_length + self.dest_state_length, self.pred_state_length)))

        self.add_submodule(self.edge_type_str + '/decoder/BC/rnn_cell',
                           model_if_absent=nn.GRUCell(decoder_input_dims, self.hyperparams['dec_rnn_dim']))
        self.add_submodule(self.edge_type_str + '/decoder/BC/initial_h',
                           model_if_absent=nn.Linear(self.z_size + self.x_size, self.hyperparams['dec_rnn_dim']))

        self.add_submodule(self.edge_type_str + '/decoder/BC/proj_to_submatrix',
                           model_if_absent=nn.Linear(self.hyperparams['dec_rnn_dim'],
                                                     self.pred_state_length))

    def p_y_xz(self, mode, x, x_nr_t, y_r, n_s_t0, z_stacked, prediction_horizon,
               num_samples, num_components=1, gmm_mode=False):
        r"""
        .. math:: p_\psi(\mathbf{y}_i \mid \mathbf{x}_i, z)

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :param x_nr_t: Joint state of node and robot (if robot is in scene).
        :param y: Future tensor.
        :param y_r: Encoded future tensor.
        :param n_s_t0: Standardized current state of the node.
        :param z_stacked: Stacked latent state. [num_samples_z * num_samples_gmm, bs, latent_state]
        :param prediction_horizon: Number of prediction timesteps.
        :param num_samples: Number of samples from the latent space.
        :param num_components: Number of GMM components.
        :param gmm_mode: If True: The mode of the GMM is sampled.
        :return: GMM2D. If mode is Predict, also samples from the GMM.
        """
        ph = prediction_horizon

        z = torch.reshape(z_stacked, (-1, self.latent.z_dim))
        zx = torch.cat([z, x.repeat(num_samples * num_components, 1)], dim=1)

        cell = self.node_modules[self.edge_type_str + '/decoder/BC/rnn_cell']
        initial_h_model = self.node_modules[self.edge_type_str + '/decoder/BC/initial_h']

        initial_state = initial_h_model(zx)

        B_submatrices, logQ_submatrices = [], []

        # log_pis is [bs, ph, num_samples, num_components]
        p_log_pis = self.latent.p_dist.logits.unsqueeze(1).repeat(num_samples, ph, 1, 1)
        q_log_pis = None
        if mode != ModeKeys.PREDICT:
            q_log_pis = self.latent.q_dist.logits.unsqueeze(1).repeat(num_samples, ph, 1, 1)

        # Infer initial action state for node from current state
        a_0 = self.node_modules[self.edge_type_str + '/decoder/BC/initial_input'](n_s_t0)

        state = initial_state
        if self.hyperparams['incl_robot_node']:
            input_ = torch.cat([zx,
                                a_0.repeat(num_samples * num_components, 1),
                                x_nr_t.repeat(num_samples * num_components, 1)], dim=1)
        else:
            input_ = torch.cat([zx, a_0.repeat(num_samples * num_components, 1)], dim=1)

        for j in range(ph):
            h_state = cell(input_, state)
            B_logQ_submatrix_vector = self.node_modules[self.edge_type_str + '/decoder/BC/proj_to_submatrix'](h_state)
            B_submatrix_vector, logQ_submatrix_vector = torch.split(B_logQ_submatrix_vector,
                                                                    split_size_or_sections=[
                                                                        self.dest_pred_state_length * 2,
                                                                        self.dest_pred_state_length],
                                                                    dim=-1)

            B_submatrices.append(B_submatrix_vector / self.hyperparams['init_scale'])
            logQ_submatrices.append(logQ_submatrix_vector)

            if self.hyperparams['incl_robot_node']:
                dec_inputs = [zx, B_submatrix_vector, logQ_submatrix_vector,
                              y_r[:, j].repeat(num_samples * num_components, 1)]
            else:
                dec_inputs = [zx, B_submatrix_vector, logQ_submatrix_vector]
            input_ = torch.cat(dec_inputs, dim=1)
            state = h_state

        # B_submatrices is [num_samples, num_components, bs, ph, dest_pred_state_length, 2]
        B_submatrices = torch.stack(B_submatrices, dim=1)
        B_submatrices = torch.reshape(B_submatrices,
                                      [num_samples, num_components,
                                       -1, ph,
                                       self.dest_pred_state_length,
                                       2]).permute(2, 3, 0, 1, 4, 5)

        # logQ_submatrices is [num_samples, num_components, bs, ph, dest_pred_state_length]
        logQ_submatrices = torch.stack(logQ_submatrices, dim=1)
        logQ_submatrices = torch.reshape(logQ_submatrices,
                                         [num_samples, num_components,
                                          -1, ph,
                                          self.dest_pred_state_length]).permute(2, 3, 0, 1, 4)

        # log_pis is now [bs, ph, num_samples, num_components]
        # B_submatrices is now [bs, ph, num_samples, num_components, dest_pred_state_length, 2]
        # logQ_submatrices is now [bs, ph, num_samples, num_components, dest_pred_state_length]
        return p_log_pis, q_log_pis, B_submatrices, logQ_submatrices

    def decoder(self, mode, x, x_nr_t, y, y_r, n_s_t0, z, prediction_horizon, num_samples):
        """
        Decoder of the CVAE.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :param x: Input / Condition tensor.
        :param x_nr_t: Joint state of node and robot (if robot is in scene).
        :param y: Future tensor.
        :param y_r: Encoded future tensor.
        :param n_s_t0: Standardized current state of the node.
        :param z: Stacked latent state.
        :param prediction_horizon: Number of prediction timesteps.
        :param num_samples: Number of samples from the latent space.
        :return: Log probability of y over p.
        """
        num_components = self.hyperparams['N'] * self.hyperparams['K']
        p_log_pis, q_log_pis, B_submatrices, Q_submatrices = self.p_y_xz(mode, x, x_nr_t, y_r, n_s_t0, z,
                                                                         prediction_horizon, num_samples,
                                                                         num_components=num_components)
        return p_log_pis, q_log_pis, B_submatrices, Q_submatrices

    def train_forward(self, dest_inputs, dest_inputs_st, first_history_indices, dest_labels, dest_labels_st,
                      orig_inputs,
                      robot, map, prediction_horizon) -> (torch.Tensor, torch.Tensor, torch.Tensor,
                                                          torch.Tensor, torch.Tensor):
        """
        Calculates the training loss for a batch.

        :param dest_inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param dest_inputs_st: Standardized input tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param dest_labels: Label tensor including the label output for each agent over time [bs, t, pred_state].
        :param dest_labels_st: Standardized label tensor.
        :param orig_inputs: Preprocessed dict (indexed by edge type) of list of neighbor states over time.
                            [[bs, t, neighbor state]]
        :param robot: Standardized robot state over time. [bs, t, robot_state]
        :param map: Tensor of Map information. [bs, channels, x, y]
        :param prediction_horizon: Number of prediction timesteps.
        :return: Scalar tensor -> nll loss
        """
        mode = ModeKeys.TRAIN

        x, x_nr_t, y_e, y_r, y, n_s_t0 = self.obtain_encoded_tensors(mode=mode, dest_inputs=dest_inputs,
                                                                     dest_inputs_st=dest_inputs_st,
                                                                     dest_labels=dest_labels,
                                                                     dest_labels_st=dest_labels_st,
                                                                     first_history_indices=first_history_indices,
                                                                     orig_history_st=orig_inputs,
                                                                     robot=robot,
                                                                     map=map)

        z, kl = self.encoder(mode, x, y_e)
        p_log_pis, q_log_pis, B_submatrices, Q_submatrices = self.decoder(mode, x, x_nr_t, y, y_r, n_s_t0, z,
                                                                          prediction_horizon,
                                                                          self.hyperparams['k'])

        if self.log_writer is not None:
            if self.hyperparams['log_histograms']:
                self.latent.summarize_for_tensorboard(self.log_writer, str(self.edge_type_str), self.curr_iter)

        return kl, p_log_pis, q_log_pis, B_submatrices, Q_submatrices

    def predict(self,
                dest_inputs,
                dest_inputs_st,
                first_history_indices,
                dest_labels,
                dest_labels_st,
                orig_inputs,
                robot,
                map,
                prediction_horizon,
                num_samples,
                z_mode=False,
                gmm_mode=False,
                full_dist=True,
                all_z_sep=False):
        """
        Predicts the future of a batch of nodes.

        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param neighbors: Preprocessed dict (indexed by edge type) of list of neighbor states over time.
                            [[bs, t, neighbor state]]
        :param neighbors_edge_value: Preprocessed edge values for all neighbor nodes [[N]]
        :param robot: Standardized robot state over time. [bs, t, robot_state]
        :param map: Tensor of Map information. [bs, channels, x, y]
        :param prediction_horizon: Number of prediction timesteps.
        :param num_samples: Number of samples from the latent space.
        :param z_mode: If True: Select the most likely latent state.
        :param gmm_mode: If True: The mode of the GMM is sampled.
        :param all_z_sep: Samples each latent mode individually without merging them into a GMM.
        :param full_dist: Samples all latent states and merges them into a GMM as output.
        :return:
        """
        mode = ModeKeys.PREDICT

        x, x_nr_t, _, y_r, _, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                                   dest_inputs=dest_inputs,
                                                                   dest_inputs_st=dest_inputs_st,
                                                                   dest_labels=dest_labels,
                                                                   dest_labels_st=dest_labels_st,
                                                                   first_history_indices=first_history_indices,
                                                                   orig_history_st=orig_inputs,
                                                                   robot=robot,
                                                                   map=map)

        self.latent.p_dist = self.p_z_x(mode, x)
        z, num_samples, num_components = self.latent.sample_p(num_samples,
                                                              mode,
                                                              most_likely_z=z_mode,
                                                              full_dist=full_dist,
                                                              all_z_sep=all_z_sep)

        p_log_pis, _, B_submatrices, logQ_submatrices = self.p_y_xz(mode, x, x_nr_t, y_r, n_s_t0, z,
                                                                    prediction_horizon,
                                                                    num_samples,
                                                                    num_components,
                                                                    gmm_mode)

        return p_log_pis, B_submatrices, logQ_submatrices
