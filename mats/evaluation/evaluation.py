import torch
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import binary_dilation
from scipy.stats import gaussian_kde
from utils import prediction_output_to_trajectories
import visualization

def compute_ade(predicted_trajs, gt_traj):
    error = np.linalg.norm(predicted_trajs - gt_traj, axis=-1)
    ade = np.mean(error, axis=-1)
    return ade.flatten()


def compute_fde(predicted_trajs, gt_traj):
    final_error = np.linalg.norm(predicted_trajs[:, -1] - gt_traj[-1], axis=-1)
    return final_error.flatten()


def compute_nll(predicted_trajs, gt_traj):
    gt_traj_t = torch.tensor(gt_traj, dtype=torch.float32)
    nll_per_t = -predicted_trajs.position_log_prob(gt_traj_t.unsqueeze(1)).numpy()
    return np.mean(nll_per_t)


def compute_obs_violations(predicted_trajs, map):
    obs_map = map.data

    interp_obs_map = RectBivariateSpline(range(obs_map.shape[1]),
                                         range(obs_map.shape[0]),
                                         binary_dilation(obs_map.T, iterations=4),
                                         kx=1, ky=1)

    old_shape = predicted_trajs.shape
    pred_trajs_map = map.to_map_points(predicted_trajs.reshape((-1, 2)))

    traj_obs_values = interp_obs_map(pred_trajs_map[:, 0], pred_trajs_map[:, 1], grid=False)
    traj_obs_values = traj_obs_values.reshape((old_shape[0], old_shape[1]))
    num_viol_trajs = np.sum(traj_obs_values.max(axis=1) > 0, dtype=float)

    return num_viol_trajs


def compute_mink_ade(predicted_trajs, gt_traj):
    ades_per_k = list()
    for k in range(1, predicted_trajs.shape[0] + 1):
        ades = compute_ade(predicted_trajs[:k], gt_traj)
        ades_per_k.append(np.min(ades))
    return ades_per_k


def compute_mink_fde(predicted_trajs, gt_traj):
    fdes_per_k = list()
    for k in range(1, predicted_trajs.shape[0] + 1):
        fdes = compute_fde(predicted_trajs[:k], gt_traj)
        fdes_per_k.append(np.min(fdes))
    return fdes_per_k


def compute_mintopk_statistics(prediction_output_dict,
                               max_hl,
                               ph,
                               node_type_enum,
                               prune_ph_to_future=False):
    (prediction_dict,
     _,
     futures_dict) = prediction_output_to_trajectories(prediction_output_dict,
                                                       max_hl,
                                                       ph,
                                                       prune_ph_to_future=prune_ph_to_future)

    batch_error_dict = dict()
    for node_type in node_type_enum:
        batch_error_dict[node_type.name] = {'min_ade_k': list(),
                                            'min_fde_k': list()}

    for t in prediction_dict.keys():
        for node in prediction_dict[t].keys():
            node_type_name = node.type.name

            gaussian_means = prediction_dict[t][node].component_distribution.mean[:, 0, :, :2]
            component_pis = prediction_dict[t][node].pis

            rank_order = torch.argsort(component_pis, descending=True)
            ranked_predictions = torch.transpose(gaussian_means, 0, 1)[rank_order]

            min_ade_errors = compute_mink_ade(ranked_predictions, futures_dict[t][node])
            min_fde_errors = compute_mink_fde(ranked_predictions, futures_dict[t][node])

            batch_error_dict[node_type_name]['min_ade_k'].append(np.array(min_ade_errors))
            batch_error_dict[node_type_name]['min_fde_k'].append(np.array(min_fde_errors))

    return batch_error_dict


def plot_mintopk_curves(mintopk_errors,
                        log_writer,
                        namespace,
                        curr_iter):
    for node_type in mintopk_errors[0].keys():
        for metric in mintopk_errors[0][node_type].keys():
            metric_batch_error = []
            for batch_errors in mintopk_errors:
                metric_batch_error.extend(batch_errors[node_type][metric])

            if len(metric_batch_error) > 0:
                mink_errors = np.stack(metric_batch_error, axis=0)
                avg_mink_errors = np.mean(mink_errors, axis=0)

                fig = visualization.visualize_mink(avg_mink_errors)
                log_writer.add_figure(f"{namespace}/{node_type}/{metric}",
                                      fig, curr_iter)


def compute_batch_statistics(prediction_output_dict,
                             max_hl,
                             ph,
                             node_type_enum,
                             kde=False,
                             obs=False,
                             map=None,
                             prune_ph_to_future=False):
    (prediction_dict,
     _,
     futures_dict) = prediction_output_to_trajectories(prediction_output_dict,
                                                       max_hl,
                                                       ph,
                                                       prune_ph_to_future=prune_ph_to_future)

    batch_error_dict = dict()
    for node_type in node_type_enum:
        batch_error_dict[node_type.name] = {'mm_ade': list(),
                                            'mm_fde': list()}
        if kde:
            batch_error_dict[node_type.name]['nll'] = list()
        if obs:
            batch_error_dict[node_type.name]['obs_viols'] = list()

    for t in prediction_dict.keys():
        for node in prediction_dict[t].keys():
            node_type_name = node.type.name
            mm_prediction = prediction_dict[t][node].mode_mode().unsqueeze(0)
            mm_ade_errors = compute_ade(mm_prediction, futures_dict[t][node])
            mm_fde_errors = compute_fde(mm_prediction, futures_dict[t][node])
            if kde:
                nll = compute_nll(prediction_dict[t][node], futures_dict[t][node])
            if obs:
                obs_viols = compute_obs_violations(prediction_dict[t][node], map)

            batch_error_dict[node_type_name]['mm_ade'].extend(list(mm_ade_errors))
            batch_error_dict[node_type_name]['mm_fde'].extend(list(mm_fde_errors))
            if kde:
                batch_error_dict[node_type_name]['nll'].append(nll)
            if obs:
                batch_error_dict[node_type_name]['obs_viols'].append(obs_viols)

    return batch_error_dict


def log_batch_errors(batch_errors_list, log_writer, namespace, curr_iter, bar_plot=[], box_plot=[]):
    for node_type in batch_errors_list[0].keys():
        for metric in batch_errors_list[0][node_type].keys():
            metric_batch_error = []
            for batch_errors in batch_errors_list:
                metric_batch_error.extend(batch_errors[node_type][metric])

            if len(metric_batch_error) > 0:
                log_writer.add_histogram(f"{namespace}/{node_type}/{metric}",
                                         metric_batch_error, curr_iter)
                log_writer.add_scalar(f"{namespace}/{node_type}/{metric}_mean",
                                      np.mean(metric_batch_error), curr_iter)
                log_writer.add_scalar(f"{namespace}/{node_type}/{metric}_median",
                                      np.median(metric_batch_error), curr_iter)


def print_batch_errors(batch_errors_list, namespace, curr_iter):
    for node_type in batch_errors_list[0].keys():
        for metric in batch_errors_list[0][node_type].keys():
            metric_batch_error = []
            for batch_errors in batch_errors_list:
                metric_batch_error.extend(batch_errors[node_type][metric])

            if len(metric_batch_error) > 0:
                print(f"{curr_iter}: {node_type.name}/{namespace}/{metric}_mean", np.mean(metric_batch_error))
                print(f"{curr_iter}: {node_type.name}/{namespace}/{metric}_median", np.median(metric_batch_error))
