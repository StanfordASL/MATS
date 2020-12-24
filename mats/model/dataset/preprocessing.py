import torch
import numpy as np
import collections.abc
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate
import dill
from environment.node import MultiNode

container_abcs = collections.abc


def restore(data):
    """
    In case we dilled some structures to share between multiple process this function will restore them.
    If the data input are not bytes we assume it was not dilled in the first place

    :param data: Possibly dilled data structure
    :return: Un-dilled data structure
    """
    if type(data) is bytes:
        return dill.loads(data)
    return data


def collate(batch):
    if len(batch) == 0:
        return batch

    elem = batch[0]
    if elem is None:
        return None

    elif isinstance(elem, container_abcs.Sequence):
        if len(elem) == 4:  # We assume those are the maps, map points, headings and patch_size
            scene_map, scene_pts, heading_angle, patch_size = zip(*batch)
            if heading_angle[0] is None:
                heading_angle = None
            else:
                heading_angle = torch.Tensor(heading_angle)
            map = scene_map[0].get_cropped_maps_from_scene_map_batch(scene_map,
                                                                     scene_pts=torch.Tensor(scene_pts),
                                                                     patch_size=patch_size[0],
                                                                     rotation=heading_angle)
            return map
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]

    elif isinstance(elem, container_abcs.Mapping):
        return {key: collate([d[key] for d in batch]) for key in elem}

    elif isinstance(elem, torch.Tensor):
        batch_shapes = set([el.shape for el in batch])
        if len(batch_shapes) != 1:
            new_batch = list(batch)
            # This is the case when we have different agent_count or largest_state_dim.
            max_agent_count = max(batch_shapes, key=lambda batch_shape: batch_shape[0])[0]
            max_largest_state_dim = max(batch_shapes, key=lambda batch_shape: batch_shape[-1])[-1]
            for idx in range(len(batch)):
                # num_nodes, time_length, largest_state_dim
                curr_shape = batch[idx].shape
                if len(curr_shape) == 3:
                    pad = (0, max_largest_state_dim - curr_shape[-1],
                           0, 0,
                           0, max_agent_count - curr_shape[0])
                    const = np.nan
                elif len(curr_shape) == 2:
                    pad = (0, max_largest_state_dim - curr_shape[-1],
                           0, 0)
                    const = np.nan
                else:
                    pad = (0, max_largest_state_dim - curr_shape[-1])
                    const = -1
                new_batch[idx] = F.pad(batch[idx], pad, "constant", const)
            return default_collate(new_batch)

    return default_collate(batch)


def get_relative_robot_traj(env, state, node_traj, robot_traj, node_type, robot_type):
    # TODO: We will have to make this more generic if robot_type != node_type
    # Make Robot State relative to node
    _, std = env.get_standardize_params(state[robot_type], node_type=robot_type)
    std[0:2] = env.attention_radius[(node_type, robot_type)]
    robot_traj_st = env.standardize(robot_traj,
                                    state[robot_type],
                                    node_type=robot_type,
                                    mean=node_traj,
                                    std=std)
    robot_traj_st_t = torch.tensor(robot_traj_st, dtype=torch.float)

    return robot_traj_st_t


def get_timestep_data(env, scene, t, state, pred_state, max_ht, max_ft, hyperparams):
    """
    Pre-processes the data for a single batch element: all node states over time for a specific time
    in a specific scene.

    :param env: Environment
    :param scene: Scene
    :param t: Timestep in scene
    :param state: Specification of the node state
    :param pred_state: Specification of the prediction state
    :param max_ht: Maximum history timesteps
    :param max_ft: Maximum future timesteps (prediction horizon)
    :param hyperparams: Model hyperparameters
    :return: Batch Element
    """

    # Node
    timestep_range_x = np.array([t - max_ht, t])
    timestep_range_y = np.array([t + 1, t + max_ft])

    nodes = scene.present_nodes(np.array([t]),
                                min_history_timesteps=1,
                                min_future_timesteps=max_ft,
                                return_robot=hyperparams['incl_robot_node'])[t]
    nodes = sorted(nodes, key=lambda node: int(node.id))  # This will make some operations later easier.
    num_nodes = len(nodes)
    num_neighbors = num_nodes - 1
    num_offdiag_A_elems = num_nodes * num_neighbors

    state_dims = {node_type: int(np.sum([len(entity_dims) for entity_dims in state[node_type].values()]))
                  for node_type in env.NodeType}
    pred_state_dims = {node_type: int(np.sum([len(entity_dims) for entity_dims in pred_state[node_type].values()]))
                  for node_type in env.NodeType}
    largest_state_dim = max(state_dims.values())

    # Preallocating data arrays
    num_agents = num_nodes
    state_lengths_in_order = torch.tensor([pred_state_dims[node.type] for node in nodes], dtype=torch.long)
    agent_types_in_order = torch.tensor([node.type.value for node in nodes], dtype=torch.long)
    robot_index = nodes.index(scene.robot)
    first_history_index = torch.empty((num_offdiag_A_elems,), dtype=torch.long)
    state_lengths = torch.empty((num_offdiag_A_elems,), dtype=torch.long)
    dest_indices = torch.empty((num_offdiag_A_elems,), dtype=torch.long)
    orig_indices = torch.empty((num_offdiag_A_elems,), dtype=torch.long)
    dest_types = torch.empty((num_offdiag_A_elems,), dtype=torch.long)
    orig_types = torch.empty((num_offdiag_A_elems,), dtype=torch.long)
    dest_inputs_t = torch.full((num_offdiag_A_elems, max_ht + 1, largest_state_dim), np.nan, dtype=torch.float)
    dest_labels_t = torch.full((num_offdiag_A_elems, max_ft, largest_state_dim), np.nan, dtype=torch.float)
    dest_inputs_st_t = torch.full((num_offdiag_A_elems, max_ht + 1, largest_state_dim), np.nan, dtype=torch.float)
    orig_inputs_st_t = torch.full((num_offdiag_A_elems, max_ht + 1, largest_state_dim), np.nan, dtype=torch.float)
    dest_labels_st_t = torch.full((num_offdiag_A_elems, max_ft, largest_state_dim), np.nan, dtype=torch.float)
    robot_traj_st_t = torch.full((num_offdiag_A_elems, max_ft + 1, state_dims[env.robot_type]), np.nan,
                                 dtype=torch.float)
    x_inits = list()
    prev_x_inits = list()
    y_targets = list()

    if type(scene.robot) is MultiNode:
        actual_node = scene.robot.get_node_at_timesteps([t, t])
        robot_future_actions_st = actual_node.get(np.array([t, t + max_ft - 1]),
                                                  hyperparams['control_state'][scene.robot.type])
    else:
        robot_future_actions_st = scene.robot.get(np.array([t, t + max_ft - 1]),
                                                  hyperparams['control_state'][scene.robot.type])

    robot_future_actions_st = env.standardize(robot_future_actions_st,
                                              hyperparams['control_state'][env.robot_type],
                                              env.robot_type)
    robot_future_actions_st_t = torch.tensor(robot_future_actions_st, dtype=torch.float)
    map_tuples = None
    if hyperparams['use_map_encoding']:
        map_tuples = list()

    edge_idx = 0
    for idx, dest_node in enumerate(nodes):
        state_length = state_dims[dest_node.type]
        if type(dest_node) is MultiNode:
            actual_node = dest_node.get_node_at_timesteps([t, t])
            x = actual_node.get(timestep_range_x, state[dest_node.type])
            y = actual_node.get(timestep_range_y, state[dest_node.type])
        else:
            x = dest_node.get(timestep_range_x, state[dest_node.type])
            y = dest_node.get(timestep_range_y, state[dest_node.type])

        first_history_index[idx * num_neighbors: (idx + 1) * num_neighbors] = (
                max_ht - dest_node.history_points_at(t)).clip(0)
        state_lengths[idx * num_neighbors: (idx + 1) * num_neighbors] = state_length
        dest_indices[idx * num_neighbors: (idx + 1) * num_neighbors] = nodes.index(dest_node)
        dest_types[idx * num_neighbors: (idx + 1) * num_neighbors] = dest_node.type.value

        _, std = env.get_standardize_params(state[dest_node.type], dest_node.type)
        std[0:2] = env.attention_radius[(dest_node.type, dest_node.type)]
        rel_state = np.zeros_like(x[0])
        rel_state[0:2] = np.array(x)[-1, 0:2]
        x_st = env.standardize(x, state[dest_node.type], dest_node.type, mean=rel_state, std=std)
        y_st = env.standardize(y, state[dest_node.type], dest_node.type, mean=rel_state)

        prev_x_init = dest_node.get(np.array([t-1]), pred_state[dest_node.type])
        x_init = dest_node.get(np.array([t]), pred_state[dest_node.type])
        if hasattr(env.NodeType, 'VEHICLE') and dest_node.type == env.NodeType.VEHICLE:
            # For vehicles with dynamically-extended unicycle models, get their heading from
            # velocity rather than the annotation heading.
            velocity = dest_node.get(np.array([t]), {'velocity': ['x', 'y']})
            x_init[:, 2] = np.arctan2(velocity[..., 1], velocity[..., 0])

            velocity = dest_node.get(np.array([t-1]), {'velocity': ['x', 'y']})
            prev_x_init[:, 2] = np.arctan2(velocity[..., 1], velocity[..., 0])

        prev_x_inits.append(torch.tensor(prev_x_init, dtype=torch.float).squeeze())
        x_inits.append(torch.tensor(x_init, dtype=torch.float).squeeze())

        y_target = dest_node.get(np.array(timestep_range_y), pred_state[dest_node.type])
        y_targets.append(torch.tensor(y_target, dtype=torch.float))

        dest_inputs_t[idx * num_neighbors: (idx + 1) * num_neighbors, :, :state_length] = torch.tensor(x, dtype=torch.float)
        dest_labels_t[idx * num_neighbors: (idx + 1) * num_neighbors, :, :state_length] = torch.tensor(y, dtype=torch.float)
        dest_inputs_st_t[idx * num_neighbors: (idx + 1) * num_neighbors, :, :state_length] = torch.tensor(x_st, dtype=torch.float)
        dest_labels_st_t[idx * num_neighbors: (idx + 1) * num_neighbors, :, :state_length] = torch.tensor(y_st, dtype=torch.float)

        # Edge
        if hyperparams['edge_encoding']:
            for jdx, orig_node in enumerate(nodes):
                if idx == jdx:
                    # Dynamics cover the diagonal, not learning.
                    continue

                orig_indices[edge_idx] = nodes.index(orig_node)
                orig_types[edge_idx] = orig_node.type.value

                if orig_node.type != dest_node.type:
                    common_type = env.NodeType.PEDESTRIAN
                    common_state = hyperparams['state'][common_type]

                    # For cases with different node types (e.g., ped-veh), we revert to relative pos, vel, acc.
                    neighbor_state_np = orig_node.get(timestep_range_x,
                                                      common_state,
                                                      padding=0.0)
                    our_state_np = dest_node.get(timestep_range_x,
                                                 common_state)

                    # Make State relative to node where neighbor and node have same state
                    _, std = env.get_standardize_params(common_state, node_type=common_type)
                    std[0:2] = env.attention_radius[(orig_node.type, dest_node.type)]
                    rel_state = np.zeros_like(neighbor_state_np) + our_state_np[-1]
                    neighbor_state_np_st = env.standardize(neighbor_state_np,
                                                           common_state,
                                                           node_type=common_type,
                                                           mean=rel_state,
                                                           std=std)

                    orig_inputs_st_t[edge_idx, :, :state_dims[common_type]] = torch.tensor(neighbor_state_np_st,
                                                                                           dtype=torch.float)

                else:
                    neighbor_state_np = orig_node.get(timestep_range_x,
                                                      state[orig_node.type],
                                                      padding=0.0)

                    # Make State relative to node where neighbor and node have same state
                    _, std = env.get_standardize_params(state[orig_node.type], node_type=orig_node.type)
                    std[0:2] = env.attention_radius[(orig_node.type, dest_node.type)]
                    rel_state = np.zeros_like(neighbor_state_np) + x[-1]
                    neighbor_state_np_st = env.standardize(neighbor_state_np,
                                                           state[orig_node.type],
                                                           node_type=orig_node.type,
                                                           mean=rel_state,
                                                           std=std)

                    orig_inputs_st_t[edge_idx, :, :state_dims[orig_node.type]] = torch.tensor(neighbor_state_np_st,
                                                                                              dtype=torch.float)

                edge_idx += 1

        # Robot
        timestep_range_r = np.array([t, t + max_ft])
        if hyperparams['incl_robot_node']:
            if type(dest_node) is MultiNode:
                actual_node = dest_node.get_node_at_timesteps([t, t])
            else:
                actual_node = dest_node

            if scene.non_aug_scene is not None:
                robot = scene.get_node_by_id(scene.non_aug_scene.robot.id)
            else:
                robot = scene.robot
            robot_type = robot.type

            if type(robot) is MultiNode:
                robot = robot.get_node_at_timesteps([t, t])

            if robot_type != dest_node.type:
                # For cases with different node types (e.g., ped-veh), we revert to relative pos, vel, acc.
                common_type = env.NodeType.PEDESTRIAN
                common_state = hyperparams['state'][common_type]

                x_node = actual_node.get(timestep_range_r, common_state)
                robot_traj = robot.get(timestep_range_r, common_state, padding=0.0)

                robot_traj_st_t[idx * num_neighbors:
                                (idx + 1) * num_neighbors] = get_relative_robot_traj(env,
                                                                                     state,
                                                                                     x_node,
                                                                                     robot_traj,
                                                                                     common_type,
                                                                                     common_type).unsqueeze(0).repeat(
                    num_neighbors, 1, 1)

            else:
                x_node = actual_node.get(timestep_range_r, state[dest_node.type])
                robot_traj = robot.get(timestep_range_r, state[robot_type], padding=0.0)

                robot_traj_st_t[idx * num_neighbors:
                                (idx + 1) * num_neighbors] = get_relative_robot_traj(env,
                                                                                     state,
                                                                                     x_node,
                                                                                     robot_traj,
                                                                                     dest_node.type,
                                                                                     robot_type).unsqueeze(0).repeat(
                    num_neighbors, 1, 1)

        # Map
        if hyperparams['use_map_encoding']:
            if dest_node.type in hyperparams['map_encoder']:
                if dest_node.non_aug_node is not None:
                    x = dest_node.non_aug_node.get(np.array([t]), state[dest_node.type])
                me_hyp = hyperparams['map_encoder'][dest_node.type]
                if 'heading_state_index' in me_hyp:
                    heading_state_index = me_hyp['heading_state_index']
                    # We have to rotate the map in the opposit direction of the agent to match them
                    if type(heading_state_index) is list:  # infer from velocity or heading vector
                        heading_angle = -np.arctan2(x[-1, heading_state_index[1]],
                                                    x[-1, heading_state_index[0]]) * 180 / np.pi
                    else:
                        heading_angle = -x[-1, heading_state_index] * 180 / np.pi
                else:
                    heading_angle = None

                scene_map = scene.map[dest_node.type]
                map_point = x[-1, :2]

                patch_size = hyperparams['map_encoder'][dest_node.type]['patch_size']
                map_tuples.extend([(scene_map, map_point, heading_angle, patch_size)] * num_neighbors)

    # Splitting it up by type.
    node_type_count_dict = {node_type.value: 0 for node_type in env.NodeType}
    edge_type_count_dict = dict()
    first_history_index_dict = dict()
    state_lengths_dict = dict()
    dest_indices_dict = dict()
    orig_indices_dict = dict()
    dest_inputs_t_dict = dict()
    dest_labels_t_dict = dict()
    dest_inputs_st_t_dict = dict()
    orig_inputs_st_t_dict = dict()
    dest_labels_st_t_dict = dict()
    robot_traj_st_t_dict = dict()
    map_tuples_dict = None
    if hyperparams['use_map_encoding']:
        map_tuples_dict = dict()

    x_inits = torch.cat(x_inits, dim=0)
    prev_x_inits = torch.cat(prev_x_inits, dim=0)
    y_targets = torch.cat(y_targets, dim=-1)
    for (orig_type, dest_type) in env.get_edge_types():
        edge_tuple = (orig_type.value, dest_type.value)

        idxs = (orig_types == orig_type.value) & (dest_types == dest_type.value)

        edge_type_count_dict[edge_tuple] = np.count_nonzero(idxs)
        first_history_index_dict[edge_tuple] = first_history_index[idxs]
        state_lengths_dict[edge_tuple] = state_lengths[idxs]
        dest_indices_dict[edge_tuple] = dest_indices[idxs]
        orig_indices_dict[edge_tuple] = orig_indices[idxs]
        dest_inputs_t_dict[edge_tuple] = dest_inputs_t[idxs][..., :state_dims[dest_type]]
        dest_labels_t_dict[edge_tuple] = dest_labels_t[idxs][..., :state_dims[dest_type]]
        dest_inputs_st_t_dict[edge_tuple] = dest_inputs_st_t[idxs][..., :state_dims[dest_type]]
        orig_inputs_st_t_dict[edge_tuple] = orig_inputs_st_t[idxs][..., :state_dims[orig_type]]
        dest_labels_st_t_dict[edge_tuple] = dest_labels_st_t[idxs][..., :state_dims[dest_type]]
        robot_traj_st_t_dict[edge_tuple] = robot_traj_st_t[idxs][..., :state_dims[env.robot_type]]
        if hyperparams['use_map_encoding']:
            map_tuples_dict[edge_tuple] = map_tuples[idxs]

    for node in nodes:
        if node.is_robot:
            continue

        node_type_count_dict[node.type.value] += 1

    return (num_agents, state_lengths_in_order, agent_types_in_order,
            robot_index, node_type_count_dict, edge_type_count_dict,
            first_history_index_dict, state_lengths_dict,
            dest_indices_dict, orig_indices_dict,
            dest_inputs_t_dict, dest_labels_t_dict,
            dest_inputs_st_t_dict, orig_inputs_st_t_dict,
            dest_labels_st_t_dict,
            robot_traj_st_t_dict, map_tuples_dict,
            x_inits, prev_x_inits, robot_future_actions_st_t, y_targets)


def get_timesteps_data(env, scene, t, state, pred_state, min_ht, max_ht, min_ft, max_ft, hyperparams):
    """
    Puts together the inputs for ALL nodes in a given scene and timestep in it.

    :param env: Environment
    :param scene: Scene
    :param t: Timestep in scene
    :param node_type: Node Type of nodes for which the data shall be pre-processed
    :param state: Specification of the node state
    :param pred_state: Specification of the prediction state
    :param edge_types: List of all Edge Types for which neighbors are pre-processed
    :param max_ht: Maximum history timesteps
    :param max_ft: Maximum future timesteps (prediction horizon)
    :param hyperparams: Model hyperparameters
    :return:
    """
    batch = list()
    out_timesteps = list()
    for timestep in t:
        out_timesteps.append(timestep)
        batch.append(get_timestep_data(env, scene, timestep, state, pred_state, max_ht, max_ft, hyperparams))

    if len(out_timesteps) == 0:
        return None
    return collate(batch), out_timesteps
