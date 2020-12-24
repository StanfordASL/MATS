import sys
import os
import pandas as pd
import dill

sys.path.append("../../mats")
from environment import Environment, Scene, Node
from utils import maybe_makedirs

dt = 0.1

standardization = {
    'PARTICLE': {
        'position': {
            'x': {'mean': 0, 'std': 5},
            'y': {'mean': 0, 'std': 5}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 2},
            'y': {'mean': 0, 'std': 2}
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        }
    }
}

maybe_makedirs('../processed')
data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])
for desired_source in ['particles']:
    dataset = dict()
    for data_class in ['train', 'val', 'test']:
        with open(data_class + '_data_2_robot.pkl', 'rb') as f:
            dataset[data_class] = dill.load(f, encoding='latin1')

        env = Environment(node_type_list=['PARTICLE'], standardization=standardization)
        attention_radius = dict()
        attention_radius[(env.NodeType.PARTICLE, env.NodeType.PARTICLE)] = 10.0
        env.attention_radius = attention_radius
        env.robot_type = env.NodeType.PARTICLE
        scenes = []
        data_dict_path = os.path.join('../processed', '_'.join([desired_source, data_class]) + '_2_robot.pkl')
        # open dataset
        for scenario in dataset[data_class]:
            max_timesteps = len(scenario[0][('position', 'x')])
            scene = Scene(timesteps=max_timesteps, dt=dt, name=desired_source + "_" + data_class)
            for node_id, data_dict in enumerate(scenario):
                node_data = pd.DataFrame(data_dict, columns=data_columns)
                node = Node(node_type=env.NodeType.PARTICLE, node_id=node_id, data=node_data)
                node.first_timestep = 0

                if node_id == 0:
                    node.is_robot = True
                    scene.robot = node

                scene.nodes.append(node)

            scenes.append(scene)
        print(f'Processed {len(scenes):.2f} scene for data class {data_class}')

        env.scenes = scenes

        if len(scenes) > 0:
            with open(data_dict_path, 'wb') as f:
                dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)
