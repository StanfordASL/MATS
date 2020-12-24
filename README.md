<p align="center"><img width="100%" src="img/MATS.png"/></p>

# MATS: An Interpretable Trajectory Forecasting Representation for Planning and Control #
This repository contains the code for the CoRL 2020 paper [MATS: An Interpretable Trajectory Forecasting Representation for Planning and Control](https://arxiv.org/abs/2009.07517) by Boris Ivanovic, Amine Elhafsi, Guy Rosman, Adrien Gaidon, and Marco Pavone.

## Installation ##

### Cloning ###
When cloning this repository, make sure you clone the submodules as well, with the following command:
```
git clone --recurse-submodules <repository cloning URL>
```
Alternatively, you can clone the repository as normal and then load submodules later with:
```
git submodule init # Initializing our local configuration file
git submodule update # Fetching all of the data from the submodules at the specified commits
```

### Environment Setup ###
First, we'll create a conda environment to hold the dependencies.
```
conda create --name mats python=3.6 -y
source activate mats
pip install -r requirements.txt
```

Then, since this project uses IPython notebooks, we'll install this conda environment as a kernel.
```
python -m ipykernel install --user --name mats --display-name "Python 3.6 (MATS)"
```

### Data Setup ###
#### Particle Dataset ####
We've already included preprocessed data splits for the particle dataset in this repository, you can see them in `experiments/processed/particles_*_2_robot.pkl`. In order to process them into a data format that our model can work with, execute the following.
```
cd experiments/particles
python particle_simulation.py # You can change sim parameters in Lines 235 and 254 - 258.
python generate_data.py # This takes the simulated particle data and formats it into the Env/Scenes/Nodes format used by our model.
```

#### nuScenes Dataset ####
Download the nuScenes dataset (this requires signing up on [their website](https://www.nuscenes.org/)). Note that the full dataset is very large, so if you only wish to test out the codebase and model then you can just download the nuScenes "mini" dataset which only requires around 4 GB of space. Extract the downloaded zip file's contents and place them in the `experiments/nuScenes` directory. Then, download the map expansion pack (v1.1) and copy the contents of the extracted `maps` folder into the `experiments/nuScenes/v1.0-mini/maps` folder. Finally, process them into a data format that our model can work with.
```
cd experiments/nuScenes

# For the mini nuScenes dataset, use the following and change the lines as per the comments in Lines 486 and 508
python process_data.py --data=./v1.0-mini --version="v1.0-mini" --output_path=../processed

# For the full nuScenes dataset, use the following
python process_data.py --data=./v1.0 --version="v1.0-trainval" --output_path=../processed
```
If you also want to make a version of the dataset at twice the frequency (for use with our proposed downstream planner), then simply add the `--half_dt` flag to the above commands.

Please note that the version of the nuScenes devkit we use (v1.0.8) might be out-of-sync with the devkit version that matches the current nuScenes dataset.

## Model Training ##
### Particle Dataset ###
To train a model on the particle dataset, you can execute a version of the following command from within the `mats/` directory.
```
CUDA_VISIBLE_DEVICES=0 python train.py --train_data_dict=particles_train_2_robot.pkl --eval_data_dict=particles_val_2_robot.pkl --log_dir=../experiments/particles/models --zero_R_rows --preprocess_workers=10
```

### nuScenes Dataset ###
To train a model on the nuScenes dataset, you can execute a version of the following command from within the `mats/` directory.
```
CUDA_VISIBLE_DEVICES=0 python train.py --train_data_dict=nuScenes_train_full.pkl --eval_data_dict=nuScenes_train_val_full.pkl --log_dir=../experiments/nuScenes/models --conf ../config/nuScenes.json --zero_R_rows --eval_every=10 --vis_every=1 --save_every=1 --train_epochs=500 --node_freq_mult_train --augment --batch_size=8 --batch_multiplier=4
```
What this means is to train a new MATS model which will be evaluated every 10 epochs, have a few outputs visualized in Tensorboard every 1 epoch, use the `nuScenes_train_full.pkl` file as the source of training data, and evaluate the partially-trained models on the data within `nuScenes_train_val_full.pkl`. Further options specify that we want to save trained models and Tensorboard logs to `../experiments/nuScenes/models`, run training for 500 epochs, augment the dataset with rotations (`--augment`), and use an _effective_ batch size of 32 (nominal batches of 8 which are multiplied by computing the gradients from 4 batches per weight update).

### CPU Training ###
By default, our training script assumes access to a GPU. If you want to train on a CPU, add `--device cpu` to the training command.

## Model Evaluation ##
### Particle Dataset ###
If you want to use a trained model to reproduce our paper figures or generate trajectories and plot them, you can use the `experiments/particles/Paper Plots.ipynb` notebook.

### nuScenes Dataset ###
If you want to use a trained model to reproduce our paper figures or generate trajectories and plot them, you can use the `experiments/nuScenes/Paper Plots.ipynb` notebook.

To evaluate a trained model's performance on forecasting pedestrians or vehicles, you can execute a version of the following command from within the `experiments/nuScenes` directory.
```
# Pedestrians
python evaluate.py --model ./models/models_26_Jul_2020_17_11_34_full_zeroRrows_batch8_fixed_edges --checkpoint 16 --data ../processed/nuScenes_val_full.pkl --output_path ./results/ --output_tag nuscenes_ped --node_type PEDESTRIAN --prediction_horizon 2 4 6

# Vehicles
python evaluate.py --model ./models/models_26_Jul_2020_17_11_34_full_zeroRrows_batch8_fixed_edges --checkpoint 16 --data ../processed/nuScenes_val_full.pkl --output_path ./results/ --output_tag nuscenes_veh --node_type VEHICLE --prediction_horizon 2 4 6
```
To change these commands for your own trained models, change the `--model` and `--checkpoint` argument values.

These scripts will print out metric values as well as produce csv files in the specified `--output_path` directory which can then be analyzed post-hoc.

## MATS + Model Predictive Control ##
The MPC part of this codebase (written primarily in Julia) is still under construction/beautification. Stay tuned for updates along this front!
