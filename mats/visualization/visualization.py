from utils import prediction_output_to_trajectories
from scipy import linalg
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import numpy as np
import seaborn as sns


def visualize_mink(avg_mink_errors):
    fig, ax = plt.subplots()
    ax.plot(range(1, avg_mink_errors.shape[0]+1), avg_mink_errors)
    ax.set_xlabel('# Components')
    ax.set_ylabel('Min. Displacement Error')
    return fig

def visualize_mats(As, Bs, Cs, pred_dists, state_lengths_in_order):
    figA = visualize_mat(As[0], pred_dists, state_lengths_in_order)
    figB = visualize_mat(Bs[0], pred_dists, state_lengths_in_order)
    figC = visualize_mat(Cs[0].unsqueeze(-1), pred_dists, state_lengths_in_order)

    return figA, figB, figC


def visualize_mats_time(As, Bs, Cs, pred_dists, state_lengths_in_order):
    figA = visualize_mat_time(As[0], pred_dists, state_lengths_in_order)
    figB = visualize_mat_time(Bs[0], pred_dists, state_lengths_in_order)
    figC = visualize_mat_time(Cs[0].unsqueeze(-1), pred_dists, state_lengths_in_order)

    return figA, figB, figC


def visualize_mat(mat, pred_dists, state_lengths_in_order):
    timesteps, num_samples, components = mat.shape[:3]
    random_dist = next(iter(pred_dists.values()))
    pis = random_dist.pis
    nrows = ncols = 5
    fig, axes = plt.subplots(figsize=(25, 25), nrows=nrows, ncols=ncols)
    line_locs = state_lengths_in_order.cumsum(1)
    for row in range(nrows):
        for col in range(ncols):
            component = row*ncols + col
            if component >= components:
                break

            sns.heatmap(mat[0, 0, row*ncols + col].cpu(),
                        annot=False, cbar=False, square=True,
                        vmin=-0.50, center=0.00, vmax=0.50, cmap='coolwarm',
                        fmt=".2f", ax=axes[row, col])
            axes[row, col].set_title('K = %d, P(z=K | x) = %.2f' % (component, pis[component]),
                                     fontsize=pis[component]*8 + 12)
            axes[row, col].hlines(line_locs, *(axes[row, col].get_xlim()), colors=['white'])
            axes[row, col].vlines(line_locs, *(axes[row, col].get_ylim()), colors=['white'])
            axes[row, col].axis('off')

    return fig


def visualize_mat_time(mat, pred_dists, state_lengths_in_order):
    timesteps, num_samples, components = mat.shape[:3]
    random_dist = next(iter(pred_dists.values()))
    pis = random_dist.pis
    ml_pi_idx = torch.argmax(pis).item()
    nrows = 3
    ncols = 4
    fig, axes = plt.subplots(figsize=(20, 15), nrows=nrows, ncols=ncols)
    line_locs = state_lengths_in_order.cumsum(1)
    for row in range(nrows):
        for col in range(ncols):
            timestep = row*ncols + col
            if timestep >= timesteps:
                break

            sns.heatmap(mat[timestep, 0, ml_pi_idx].cpu(),
                        annot=False, cbar=False, square=False,
                        vmin=-0.50, center=0.00, vmax=0.50, cmap='coolwarm',
                        fmt=".2f", ax=axes[row, col])
            axes[row, col].set_title('P(z = %d | x) = %.2f, t = %d' % (ml_pi_idx, pis[ml_pi_idx], timestep))
            axes[row, col].hlines(line_locs, *(axes[row, col].get_xlim()), colors=['white'])
            axes[row, col].vlines(line_locs, *(axes[row, col].get_ylim()), colors=['white'])
            axes[row, col].axis('off')

    return fig


def plot_trajectories(ax,
                      prediction_dict,
                      histories_dict,
                      futures_dict,
                      line_alpha=0.7,
                      line_width=0.2,
                      edge_width=2,
                      circle_edge_width=0.5,
                      node_circle_size=0.3,
                      batch_num=0,
                      kde=False,
                      x_min=0,
                      y_min=0,
                      **kwargs):

    offset_arr = np.array([x_min, y_min])
    cmap = ['k', 'b', 'y', 'g', 'r']

    for node in histories_dict:
        history = histories_dict[node] + offset_arr
        future = futures_dict[node] + offset_arr
        if not node.is_robot:
            predictions = prediction_dict[node]
            predictions_map_pts = None
            if isinstance(predictions, tuple):
                predictions, predictions_map_pts = predictions

        if np.isnan(history[-1]).any():
            continue

        ax.plot(history[:, 0], history[:, 1], 'k--')

        if not node.is_robot:
            pis = predictions.pis
            for component in range(predictions.mixture_distribution.param_shape[-1]):
                if 'pi_alpha' in kwargs and kwargs['pi_alpha']:
                    line_alpha = pis[component].item()

                if kde and predictions.batch_shape[1] >= 50:
                    for t in range(predictions.shape[2]):
                        sns.kdeplot(predictions[batch_num, :, t, 0], predictions[batch_num, :, t, 1],
                                    ax=ax, shade=True, shade_lowest=False,
                                    color=np.random.choice(cmap), alpha=0.8)

                # TODO: Should probably just remove samples, honestly.
                if predictions_map_pts is None:
                    ax.plot(predictions.component_distribution.mean[:, 0, component, 0] + x_min,
                            predictions.component_distribution.mean[:, 0, component, 1] + y_min,
                            color=cmap[node.type.value],
                            linewidth=line_width,
                            alpha=line_alpha)
                else:
                    ax.plot(predictions_map_pts[:, 0, component, 0],
                            predictions_map_pts[:, 0, component, 1],
                            color=cmap[node.type.value],
                            linewidth=line_width,
                            alpha=line_alpha)

        ax.plot(future[:, 0],
                future[:, 1],
                'w--',
                path_effects=[pe.Stroke(linewidth=edge_width, foreground='k'), pe.Normal()])

        # Current Node Position
        circle = plt.Circle((history[-1, 0],
                             history[-1, 1]),
                            node_circle_size,
                            facecolor='g' if not node.is_robot else 'gray',
                            edgecolor='k',
                            lw=circle_edge_width,
                            zorder=3)
        ax.add_artist(circle)

    ax.axis('equal')


def visualize_prediction(ax,
                         pred_dists,
                         dt,
                         max_hl,
                         ph,
                         robot_node=None,
                         map=None,
                         x_min=0,
                         y_min=0,
                         **kwargs):

    prediction_dict, histories_dict, futures_dict = prediction_output_to_trajectories(pred_dists,
                                                                                      max_hl,
                                                                                      ph,
                                                                                      map=map)

    assert(len(prediction_dict.keys()) <= 1)
    if len(prediction_dict.keys()) == 0:
        return
    ts_key = list(prediction_dict.keys())[0]

    prediction_dict = prediction_dict[ts_key]
    histories_dict = histories_dict[ts_key]
    futures_dict = futures_dict[ts_key]

    if robot_node is not None:
        histories_dict[robot_node] = robot_node.get(np.array([ts_key-max_hl, ts_key]), 
                                                    {'position': ['x', 'y']})
        futures_dict[robot_node] = robot_node.get(np.array([ts_key+1, ts_key+ph]), 
                                                  {'position': ['x', 'y']})

    if map is not None:
        ax.imshow(map.as_image(), origin='lower', alpha=0.5)
    plot_trajectories(ax, prediction_dict, histories_dict, futures_dict,
                      x_min=x_min, y_min=y_min, **kwargs)


def visualize_distribution(ax,
                           prediction_distribution_dict,
                           map=None,
                           pi_threshold=0.05,
                           **kwargs):
    if map is not None:
        ax.imshow(map.as_image(), origin='lower', alpha=0.5)

    for node, pred_dist in prediction_distribution_dict.items():
        if pred_dist.mus.shape[:2] != (1, 1):
            return

        means = pred_dist.mus.squeeze().cpu().numpy()
        covs = pred_dist.get_covariance_matrix().squeeze().cpu().numpy()
        pis = pred_dist.pis_cat_dist.probs.squeeze().cpu().numpy()

        for timestep in range(means.shape[0]):
            for z_val in range(means.shape[1]):
                mean = means[timestep, z_val]
                covar = covs[timestep, z_val]
                pi = pis[timestep, z_val]

                if pi < pi_threshold:
                    continue

                v, w = linalg.eigh(covar)
                v = 2. * np.sqrt(2.) * np.sqrt(v)
                u = w[0] / linalg.norm(w[0])

                # Plot an ellipse to show the Gaussian component
                angle = np.arctan(u[1] / u[0])
                angle = 180. * angle / np.pi  # convert to degrees
                ell = patches.Ellipse(mean, v[0], v[1], 180. + angle, color='blue' if node.type.name == 'VEHICLE' else 'orange')
                ell.set_edgecolor(None)
                ell.set_clip_box(ax.bbox)
                ell.set_alpha(pi/10)
                ax.add_artist(ell)
