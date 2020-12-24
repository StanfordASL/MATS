from collections import defaultdict
from torch.utils import data
import numpy as np
from environment.node import MultiNode
from .preprocessing import get_timestep_data


class EnvironmentDataset(object):
    def __init__(self, env, state, pred_state, node_freq_mult, scene_freq_mult, hyperparams, **kwargs):
        self.env = env
        self.state = state
        self.pred_state = pred_state
        self.hyperparams = hyperparams
        self.max_ht = self.hyperparams['maximum_history_length']
        self.max_ft = kwargs['min_future_timesteps']
        self._augment = False
        self.dataset = SceneTimeDataset(env, state, pred_state, node_freq_mult,
                                        scene_freq_mult, hyperparams, **kwargs)

    @property
    def augment(self):
        return self._augment

    @augment.setter
    def augment(self, value):
        self._augment = value
        self.dataset.augment = value


class SceneTimeDataset(data.Dataset):
    def __init__(self, env, state, pred_state, node_freq_mult,
                 scene_freq_mult, hyperparams, augment=False, **kwargs):
        self.env = env
        self.state = state
        self.pred_state = pred_state
        self.hyperparams = hyperparams
        self.max_ht = self.hyperparams['maximum_history_length']
        self.max_ft = kwargs['min_future_timesteps']

        self.augment = augment

        self.scene_time_dict = defaultdict(list)
        self.index = self.index_env(node_freq_mult, scene_freq_mult, **kwargs)
        self.len = len(self.index)

    def index_env(self, node_freq_mult, scene_freq_mult, **kwargs):
        index = list()
        for scene in self.env.scenes:
            present_node_dict = scene.present_nodes(np.arange(0, scene.timesteps), **kwargs)
            for t in present_node_dict:
                if type(scene.robot) is MultiNode and t not in scene.robot.timestep_node_dict:
                    continue

                if len(present_node_dict[t]) == 1:
                    continue

                # robot_actual_node = scene.robot.get_node_at_timesteps([t, t])
                # if robot_actual_node.id == 'EMPTY':
                #     continue

                not_present_enough = ((t - kwargs['min_history_timesteps'] < scene.robot.first_timestep)
                                      or (scene.robot.last_timestep < t + kwargs['min_future_timesteps']))
                if not_present_enough:
                    continue

                index += [(scene, t)] * (scene.frequency_multiplier if scene_freq_mult else 1)
                self.scene_time_dict[scene].append(t)

        return index

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        (scene, t) = self.index[i]

        if self.augment:
            scene = scene.augment()

        return get_timestep_data(self.env, scene, t, self.state, self.pred_state, self.max_ht, self.max_ft,
                                 self.hyperparams)
