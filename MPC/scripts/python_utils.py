import sys
import os
import dill
import glob
import json
import random
import torch
import numpy as np
import pandas as pd

sys.path.append("../../mats")
from model.model_registrar import ModelRegistrar
from model.mats import MATS
from environment.node import MultiNode
from model.dataset import EnvironmentDataset
import evaluation
import visualization

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def load_model(model_dir, env, ts=100):
    model_registrar = ModelRegistrar(model_dir, 'cpu')
    model_registrar.load_models(ts)
    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:

        hyperparams = json.load(config_json)
    mats = MATS(model_registrar, hyperparams, None, 'cpu')
    return mats, hyperparams

def load_data_set(filename):
    with open(filename, 'rb') as f:
        env = dill.load(f, encoding='latin1')
    return env

def calculate_scene_graphs(env, hyperparams):
    scenes = env.scenes
    for scene in scenes:
        scene.calculate_scene_graph(env.attention_radius,
                                    hyperparams['edge_addition_filter'],
                                    hyperparams['edge_removal_filter'])

def predict(mats, hyperparams, scene, timestep, num_modes):
    ph = hyperparams['prediction_horizon']

    with torch.no_grad():
        pred_dists, non_rob_rows, As, Bs, Qs, affine_terms, state_lengths_in_order = mats.predict(
                                  scene,
                                  np.array([timestep]),
                                  ph,
                                  min_future_timesteps=ph,
                                  include_B=hyperparams['include_B'],
                                  zero_R_rows=hyperparams['zero_R_rows'])

    A = As[0].numpy()
    B = Bs[0].numpy()
    Q = Qs[0].numpy()
    affine_terms = affine_terms[0].numpy()

    prediction_info = dict()

    state_lengths_in_order = state_lengths_in_order.squeeze().numpy()
    current_state_idx = state_lengths_in_order[0]
    for idx, node in enumerate(pred_dists[timestep]):
        curr_state = node.get(np.array([timestep, timestep]), hyperparams['pred_state'][node.type.name])
        node_predictions = pred_dists[timestep][node]
        state_predictions = node_predictions.component_distribution.mean.numpy()
        mode_probs = node_predictions.pis.numpy()
        rank_order = np.argsort(mode_probs)[::-1]
        node_str = '/'.join([node.type.name, str(node.id)])
        prediction_info[node_str] = {'node_type' : node.type.name,
                                     'node_idx' : idx+1,
                                     'current_state' : curr_state,
                                     'mode_probs' : mode_probs[rank_order[:num_modes]],
                                     'state_predictions' : state_predictions[:, 0, rank_order[:num_modes]],
                                     'state_uncertainties' : Q[:, 0, rank_order[:num_modes], current_state_idx:current_state_idx+state_lengths_in_order[idx+1]]}
        current_state_idx += state_lengths_in_order[idx+1]

    dynamics_dict = {'A' : A[:, 0, rank_order[:num_modes]],
                     'B' : B[:, 0, rank_order[:num_modes]],
                     'affine_terms' : affine_terms[:, 0, rank_order[:num_modes]]}

    return prediction_info, dynamics_dict
