import sys
import os
import dill
import glob
import json
import random
import argparse
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append("../../mats")
from tqdm import tqdm, trange
from model.model_registrar import ModelRegistrar
from model.mats import MATS
from environment.node import MultiNode
from model.dataset import EnvironmentDataset
import evaluation
import visualization
import helper as plotting_helper
import utils
from scipy.interpolate import RectBivariateSpline

from nuscenes.nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.map_expansion.map_api import NuScenesMap

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model full path", type=str)
parser.add_argument("--checkpoint", help="model checkpoint to evaluate", type=int)
parser.add_argument("--data", help="full path to data file", type=str)
parser.add_argument("--output_path", help="path to output csv file", type=str, default='results')
parser.add_argument("--output_tag", help="name tag for output file", type=str)
parser.add_argument("--node_type", help="node type to evaluate", type=str)
parser.add_argument("--batch_size", help="number of timesteps to evaluate in a batch", type=int, default=30)
parser.add_argument("--prediction_horizon", nargs='+', help="prediction horizon", type=int, default=None)
args = parser.parse_args()


def load_model(model_dir, env, ts=100):
    model_registrar = ModelRegistrar(model_dir, 'cpu')
    model_registrar.load_models(ts)
    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)

    mats = MATS(model_registrar, hyperparams, None, 'cpu')
    return mats, hyperparams


if __name__ == "__main__":
    with open(args.data, 'rb') as f:
        env = dill.load(f, encoding='latin1')

    eval_stg, hyperparams = load_model(args.model, env, ts=args.checkpoint)

    if 'override_attention_radius' in hyperparams:
        for attention_radius_override in hyperparams['override_attention_radius']:
            node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
            env.attention_radius[(node_type1, node_type2)] = float(attention_radius)
            
    eval_dataset = EnvironmentDataset(env,
                                      hyperparams['state'],
                                      hyperparams['pred_state'],
                                      scene_freq_mult=hyperparams['scene_freq_mult_eval'],
                                      node_freq_mult=hyperparams['node_freq_mult_eval'],
                                      hyperparams=hyperparams,
                                      min_history_timesteps=hyperparams['minimum_history_length'],
                                      min_future_timesteps=hyperparams['prediction_horizon'])

    eval_stg.set_environment(env)
    eval_stg.set_annealing_params()
    
    print("-- Preparing Node Graph")
    scenes = env.scenes
    for scene in tqdm(scenes):
        scene.calculate_scene_graph(env.attention_radius,
                                    hyperparams['edge_addition_filter'],
                                    hyperparams['edge_removal_filter'])

    for ph in args.prediction_horizon:
        print(f"Prediction Horizon: {ph}")
        max_hl = hyperparams['maximum_history_length']

        with torch.no_grad():
            eval_ade_batch_errors = np.array([])
            eval_fde_batch_errors = np.array([])
            eval_kde_nll = np.array([])

            for scene, times in tqdm(eval_dataset.dataset.scene_time_dict.items()):
                for idx_start in range(0, len(times), args.batch_size):
                    timesteps = np.array(times[idx_start:idx_start + args.batch_size], dtype=int)

                    predictions, _, _, _, _, _, _ = eval_stg.predict(scene,
                                                                     timesteps,
                                                                     ph,
                                                                     min_future_timesteps=hyperparams['prediction_horizon'],
                                                                     include_B=hyperparams['include_B'],
                                                                     zero_R_rows=hyperparams['zero_R_rows'])

                    batch_error_dict = evaluation.compute_batch_statistics(predictions,
                                                                           max_hl=max_hl,
                                                                           ph=ph,
                                                                           node_type_enum=env.NodeType,
                                                                           map=None,
                                                                           prune_ph_to_future=False,
                                                                           kde=True)

                    eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[args.node_type]['mm_ade']))
                    eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[args.node_type]['mm_fde']))
                    eval_kde_nll = np.hstack((eval_kde_nll, batch_error_dict[args.node_type]['nll']))

            print('ML ADE:', np.mean(eval_ade_batch_errors))
            print('ML FDE:', np.mean(eval_fde_batch_errors))
            print('FULL KDE:', np.mean(eval_kde_nll))
            pd.DataFrame({'value': eval_ade_batch_errors, 'metric': 'ade', 'type': 'ml'}
                         ).to_csv(os.path.join(args.output_path, args.output_tag + "_" + str(ph) + '_ade_most_likely_z.csv'))
            pd.DataFrame({'value': eval_fde_batch_errors, 'metric': 'fde', 'type': 'ml'}
                         ).to_csv(os.path.join(args.output_path, args.output_tag + "_" + str(ph) + '_fde_most_likely_z.csv'))
            pd.DataFrame({'value': eval_kde_nll, 'metric': 'nll', 'type': 'full'}
                         ).to_csv(os.path.join(args.output_path, args.output_tag + "_" + str(ph) + '_nll_full.csv'))
