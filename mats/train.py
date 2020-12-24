import torch
from torch import nn, optim, utils
import numpy as np
import os
import time
import dill
import json
import random
import pathlib
from tqdm import tqdm
import visualization
import evaluation
import matplotlib.pyplot as plt
from argument_parser import args
from model.mats import MATS
from model.model_registrar import ModelRegistrar
from model.dataset import EnvironmentDataset, collate
from tensorboardX import SummaryWriter

if not torch.cuda.is_available() or args.device == 'cpu':
    args.device = torch.device('cpu')
else:
    if torch.cuda.device_count() == 1:
        # If you have CUDA_VISIBLE_DEVICES set, which you should,
        # then this will prevent leftover flag arguments from
        # messing with the device allocation.
        args.device = 'cuda:0'

    args.device = torch.device(args.device)

if args.eval_device is None:
    args.eval_device = torch.device('cpu')

if args.device != torch.device('cpu'):
    # This is needed for memory pinning using a DataLoader (otherwise memory is pinned to cuda:0 by default)
    torch.cuda.set_device(args.device)

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def main():
    # Load hyperparameters from json
    if not os.path.exists(args.conf):
        print('Config json not found!')
    with open(args.conf, 'r', encoding='utf-8') as conf_json:
        hyperparams = json.load(conf_json)

    # Add hyperparams from arguments
    hyperparams['dynamic_edges'] = args.dynamic_edges
    hyperparams['edge_state_combine_method'] = args.edge_state_combine_method
    hyperparams['edge_influence_combine_method'] = args.edge_influence_combine_method
    hyperparams['edge_addition_filter'] = args.edge_addition_filter
    hyperparams['edge_removal_filter'] = args.edge_removal_filter
    hyperparams['batch_size'] = args.batch_size
    hyperparams['k_eval'] = args.k_eval
    hyperparams['offline_scene_graph'] = args.offline_scene_graph
    hyperparams['incl_robot_node'] = not args.no_robot_node
    hyperparams['node_freq_mult_train'] = args.node_freq_mult_train
    hyperparams['node_freq_mult_eval'] = args.node_freq_mult_eval
    hyperparams['scene_freq_mult_train'] = args.scene_freq_mult_train
    hyperparams['scene_freq_mult_eval'] = args.scene_freq_mult_eval
    hyperparams['scene_freq_mult_viz'] = args.scene_freq_mult_viz
    hyperparams['edge_encoding'] = not args.no_edge_encoding
    hyperparams['use_map_encoding'] = args.map_encoding
    hyperparams['augment'] = args.augment
    hyperparams['override_attention_radius'] = args.override_attention_radius
    hyperparams['include_B'] = not args.no_B
    hyperparams['reg_B'] = args.reg_B
    hyperparams['reg_B_weight'] = args.reg_B_weight
    hyperparams['zero_R_rows'] = args.zero_R_rows
    hyperparams['reg_A_slew'] = args.reg_A_slew
    hyperparams['reg_A_slew_weight'] = args.reg_A_slew_weight

    print('-----------------------')
    print('| TRAINING PARAMETERS |')
    print('-----------------------')
    print('| batch_size: %d' % args.batch_size)
    print('| device: %s' % args.device)
    print('| eval_device: %s' % args.eval_device)
    print('| Offline Scene Graph Calculation: %s' % args.offline_scene_graph)
    print('| EE state_combine_method: %s' % args.edge_state_combine_method)
    print('| EIE scheme: %s' % args.edge_influence_combine_method)
    print('| dynamic_edges: %s' % args.dynamic_edges)
    print('| robot node: %s' % (not args.no_robot_node))
    print('| edge_addition_filter: %s' % args.edge_addition_filter)
    print('| edge_removal_filter: %s' % args.edge_removal_filter)
    print('| MHL: %s' % hyperparams['minimum_history_length'])
    print('| PH: %s' % hyperparams['prediction_horizon'])
    print('-----------------------')

    log_writer = None
    model_dir = None
    if not args.debug:
        # Create the log and model directiory if they're not present.
        model_dir = os.path.join(args.log_dir,
                                 'models_' + time.strftime('%d_%b_%Y_%H_%M_%S', time.localtime()) + args.log_tag)
        pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)

        # Save config to model directory
        with open(os.path.join(model_dir, 'config.json'), 'w') as conf_json:
            json.dump(hyperparams, conf_json)

        log_writer = SummaryWriter(log_dir=model_dir)

    # Load training and evaluation environments and scenes
    train_data_path = os.path.join(args.data_dir, args.train_data_dict)
    with open(train_data_path, 'rb') as f:
        train_env = dill.load(f, encoding='latin1')

    for attention_radius_override in args.override_attention_radius:
        node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
        train_env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

    if train_env.robot_type is None and hyperparams['incl_robot_node']:
        train_env.robot_type = train_env.NodeType[0]  # TODO: Make more general, allow the user to specify?
        for scene in train_env.scenes:
            scene.add_robot_from_nodes(train_env.robot_type,
                                       hyperparams=hyperparams,
                                       min_timesteps=hyperparams['minimum_history_length'] + 1 + hyperparams[
                                           'prediction_horizon'])

    train_scenes = train_env.scenes
    train_dataset = EnvironmentDataset(train_env,
                                       hyperparams['state'],
                                       hyperparams['pred_state'],
                                       scene_freq_mult=hyperparams['scene_freq_mult_train'],
                                       node_freq_mult=hyperparams['node_freq_mult_train'],
                                       hyperparams=hyperparams,
                                       min_history_timesteps=hyperparams['minimum_history_length'],
                                       min_future_timesteps=hyperparams['prediction_horizon'])
    train_data_loader = utils.data.DataLoader(train_dataset.dataset,
                                              collate_fn=collate,
                                              pin_memory=False if args.device == torch.device('cpu') else True,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.preprocess_workers)

    print(f"Loaded training data from {train_data_path}")

    eval_scenes = []
    if args.eval_every is not None:
        eval_data_path = os.path.join(args.data_dir, args.eval_data_dict)
        with open(eval_data_path, 'rb') as f:
            eval_env = dill.load(f, encoding='latin1')

        for attention_radius_override in args.override_attention_radius:
            node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
            eval_env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

        if eval_env.robot_type is None and hyperparams['incl_robot_node']:
            eval_env.robot_type = eval_env.NodeType[0]  # TODO: Make more general, allow the user to specify?
            for scene in eval_env.scenes:
                scene.add_robot_from_nodes(eval_env.robot_type,
                                           hyperparams=hyperparams,
                                           min_timesteps=hyperparams['minimum_history_length'] + 1 + hyperparams[
                                               'prediction_horizon'])

        eval_scenes = eval_env.scenes
        eval_dataset = EnvironmentDataset(eval_env,
                                          hyperparams['state'],
                                          hyperparams['pred_state'],
                                          scene_freq_mult=hyperparams['scene_freq_mult_eval'],
                                          node_freq_mult=hyperparams['node_freq_mult_eval'],
                                          hyperparams=hyperparams,
                                          min_history_timesteps=hyperparams['minimum_history_length'],
                                          min_future_timesteps=hyperparams['prediction_horizon'])
        eval_data_loader = utils.data.DataLoader(eval_dataset.dataset,
                                                 collate_fn=collate,
                                                 pin_memory=False if args.eval_device == torch.device('cpu') else True,
                                                 batch_size=args.eval_batch_size,
                                                 shuffle=True,
                                                 num_workers=args.preprocess_workers)

        print(f"Loaded evaluation data from {eval_data_path}")

    # Offline Calculate Scene Graph
    if hyperparams['offline_scene_graph'] == 'yes':
        print(f"Offline calculating scene graphs")
        for i, scene in enumerate(train_scenes):
            scene.calculate_scene_graph(train_env.attention_radius,
                                        hyperparams['edge_addition_filter'],
                                        hyperparams['edge_removal_filter'])
        print(f"Created Scene Graphs for Training Scenes")

        for i, scene in enumerate(eval_scenes):
            scene.calculate_scene_graph(eval_env.attention_radius,
                                        hyperparams['edge_addition_filter'],
                                        hyperparams['edge_removal_filter'])
        print(f"Created Scene Graphs for Evaluation Scenes")

    model_registrar = ModelRegistrar(model_dir, args.device)

    mats = MATS(model_registrar,
                      hyperparams,
                      log_writer,
                      args.device)

    mats.set_environment(train_env)
    mats.set_annealing_params()
    print('Created Training Model.')

    eval_mats = None
    if args.eval_every is not None or args.vis_every is not None:
        eval_mats = MATS(model_registrar,
                               hyperparams,
                               log_writer,
                               args.eval_device)
        eval_mats.set_environment(eval_env)
        eval_mats.set_annealing_params()
    print('Created Evaluation Model.')

    optimizer = optim.Adam([{'params': model_registrar.get_all_but_name_match('map_encoder').parameters()},
                            {'params': model_registrar.get_name_match('map_encoder').parameters(), 'lr': 0.0008}],
                           lr=hyperparams['learning_rate'])
    # Set Learning Rate
    if hyperparams['learning_rate_style'] == 'const':
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)
    elif hyperparams['learning_rate_style'] == 'exp':
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=hyperparams['learning_decay_rate'])

    #################################
    #           TRAINING            #
    #################################
    curr_iter = 0
    for epoch in range(1, args.train_epochs + 1):
        model_registrar.to(args.device)
        train_dataset.augment = args.augment
        train_data_iterator = iter(train_data_loader)
        pbar = tqdm(total=len(train_data_loader), ncols=80)
        for _ in range(0, len(train_data_loader), args.batch_multiplier):
            mats.set_curr_iter(curr_iter)
            mats.step_annealers()

            # Zeroing gradients.
            optimizer.zero_grad()

            train_losses = list()
            for mb_num in range(args.batch_multiplier):
                try:
                    train_loss = mats.train_loss(next(train_data_iterator),
                                                       include_B=hyperparams['include_B'],
                                                       reg_B=hyperparams['reg_B'],
                                                       zero_R_rows=hyperparams['zero_R_rows']) / args.batch_multiplier
                    train_losses.append(train_loss.item())
                    train_loss.backward()
                except StopIteration:
                    break

            pbar.update(args.batch_multiplier)
            pbar.set_description(f"Epoch {epoch} L: {sum(train_losses):.2f}")

            # Clipping gradients.
            if hyperparams['grad_clip'] is not None:
                nn.utils.clip_grad_value_(model_registrar.parameters(), hyperparams['grad_clip'])

            # Optimizer step.
            optimizer.step()

            # Stepping forward the learning rate scheduler and annealers.
            lr_scheduler.step()

            if not args.debug:
                log_writer.add_scalar(f"train/learning_rate",
                                      lr_scheduler.get_last_lr()[0],
                                      curr_iter)
                log_writer.add_scalar(f"train/loss", sum(train_losses), curr_iter)

            curr_iter += 1

        train_dataset.augment = False
        if args.eval_every is not None or args.vis_every is not None:
            eval_mats.set_curr_iter(epoch)

        #################################
        #        VISUALIZATION          #
        #################################
        if args.vis_every is not None and not args.debug and epoch % args.vis_every == 0 and epoch > 0:
            max_hl = hyperparams['maximum_history_length']
            ph = hyperparams['prediction_horizon']
            with torch.no_grad():
                index = train_dataset.dataset.index
                rand_elem = index[random.randrange(len(index))]
                scene, timestep = rand_elem[0], np.array([rand_elem[1]])
                pred_dists, non_rob_rows, As, Bs, Qs, affine_terms, state_lengths_in_order = mats.predict(scene,
                                                                                                                timestep,
                                                                                                                ph,
                                                                                                                min_future_timesteps=ph,
                                                                                                                include_B=
                                                                                                                hyperparams[
                                                                                                                    'include_B'],
                                                                                                                zero_R_rows=
                                                                                                                hyperparams[
                                                                                                                    'zero_R_rows'])

                # Plot predicted timestep for random scene
                fig, ax = plt.subplots(figsize=(10, 10))
                visualization.visualize_prediction(ax,
                                                   pred_dists,
                                                   scene.dt,
                                                   max_hl=max_hl,
                                                   ph=ph,
                                                   map=scene.map['VISUALIZATION'] if scene.map is not None else None,
                                                   robot_node=scene.robot)
                ax.set_title(f"{scene.name}-t: {timestep}")
                log_writer.add_figure('train/all_modes', fig, epoch)

                # Plot A, B, Q matrices.
                figs = visualization.visualize_mats(As, Bs, Qs, pred_dists[timestep.item()], state_lengths_in_order)
                for idx, fig in enumerate(figs):
                    fig.suptitle(f"{scene.name}-t: {timestep}")
                    log_writer.add_figure(f'train/{"ABQ"[idx]}_mat', fig, epoch)

                # Plot most-likely A, B, Q matrices across time.
                figs = visualization.visualize_mats_time(As, Bs, Qs, pred_dists[timestep.item()],
                                                         state_lengths_in_order)
                for idx, fig in enumerate(figs):
                    fig.suptitle(f"{scene.name}-t: {timestep}")
                    log_writer.add_figure(f'train/ml_{"ABQ"[idx]}_mat', fig, epoch)

                model_registrar.to(args.eval_device)

                # Predict random timestep to plot for eval data set
                index = eval_dataset.dataset.index
                rand_elem = index[random.randrange(len(index))]
                scene, timestep = rand_elem[0], np.array([rand_elem[1]])

                pred_dists, non_rob_rows, As, Bs, Qs, affine_terms, state_lengths_in_order = eval_mats.predict(
                    scene,
                    timestep,
                    ph,
                    min_future_timesteps=ph,
                    include_B=
                    hyperparams[
                        'include_B'],
                    zero_R_rows=
                    hyperparams[
                        'zero_R_rows'])

                # Plot predicted timestep for random scene
                fig, ax = plt.subplots(figsize=(10, 10))
                visualization.visualize_prediction(ax,
                                                   pred_dists,
                                                   scene.dt,
                                                   max_hl=max_hl,
                                                   ph=ph,
                                                   map=scene.map['VISUALIZATION'] if scene.map is not None else None,
                                                   robot_node=scene.robot)
                ax.set_title(f"{scene.name}-t: {timestep}")
                log_writer.add_figure('eval/all_modes', fig, epoch)

                # Plot A, B, Q matrices.
                figs = visualization.visualize_mats(As, Bs, Qs, pred_dists[timestep.item()], state_lengths_in_order)
                for idx, fig in enumerate(figs):
                    fig.suptitle(f"{scene.name}-t: {timestep}")
                    log_writer.add_figure(f'eval/{"ABQ"[idx]}_mat', fig, epoch)

                # Plot most-likely A, B, Q matrices across time.
                figs = visualization.visualize_mats_time(As, Bs, Qs, pred_dists[timestep.item()],
                                                         state_lengths_in_order)
                for idx, fig in enumerate(figs):
                    fig.suptitle(f"{scene.name}-t: {timestep}")
                    log_writer.add_figure(f'eval/ml_{"ABQ"[idx]}_mat', fig, epoch)

        #################################
        #           EVALUATION          #
        #################################
        if args.eval_every is not None and not args.debug and epoch % args.eval_every == 0 and epoch > 0:
            max_hl = hyperparams['maximum_history_length']
            ph = hyperparams['prediction_horizon']
            model_registrar.to(args.eval_device)
            with torch.no_grad():
                # Calculate evaluation loss
                eval_losses = []
                print(f"Starting Evaluation @ epoch {epoch}")
                pbar = tqdm(eval_data_loader, ncols=80)
                for batch in pbar:
                    eval_loss = eval_mats.eval_loss(batch,
                                                          include_B=hyperparams['include_B'],
                                                          zero_R_rows=hyperparams['zero_R_rows'])
                    pbar.set_description(f"Epoch {epoch} L: {eval_loss.item():.2f}")
                    eval_losses.append({'full_state': {'nll': [eval_loss]}})
                    del batch

                evaluation.log_batch_errors(eval_losses,
                                            log_writer,
                                            'eval',
                                            epoch)

                # Predict batch timesteps for evaluation dataset evaluation
                eval_batch_errors = []
                eval_mintopk_errors = []
                for scene, times in tqdm(eval_dataset.dataset.scene_time_dict.items(),
                                         desc='Evaluation', ncols=80):
                    timesteps = np.random.choice(times, args.eval_batch_size)

                    pred_dists, non_rob_rows, As, Bs, Qs, affine_terms, _ = eval_mats.predict(scene,
                                                                                                    timesteps,
                                                                                                    ph=ph,
                                                                                                    min_future_timesteps=ph,
                                                                                                    include_B=
                                                                                                    hyperparams[
                                                                                                        'include_B'],
                                                                                                    zero_R_rows=
                                                                                                    hyperparams[
                                                                                                        'zero_R_rows'])

                    eval_batch_errors.append(evaluation.compute_batch_statistics(pred_dists,
                                                                                 max_hl=max_hl,
                                                                                 ph=ph,
                                                                                 node_type_enum=eval_env.NodeType,
                                                                                 map=None))  # scene.map))

                    eval_mintopk_errors.append(evaluation.compute_mintopk_statistics(pred_dists,
                                                                                     max_hl,
                                                                                     ph=ph,
                                                                                     node_type_enum=eval_env.NodeType))

                evaluation.log_batch_errors(eval_batch_errors,
                                            log_writer,
                                            'eval',
                                            epoch)

                evaluation.plot_mintopk_curves(eval_mintopk_errors,
                                               log_writer,
                                               'eval',
                                               epoch)

        if args.save_every is not None and args.debug is False and epoch % args.save_every == 0:
            model_registrar.save_models(epoch)


if __name__ == '__main__':
    main()
