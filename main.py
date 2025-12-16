import os
import shutil

import tensorflow as tf
import pandas     as pd

from NeuralNetwork.classes       import NeuralNetwork  , Plotter
from NeuralNetworkDriver.classes import InputDataLoader

INPUT_DIR_ADDR    = './Data/'
OUTPUT_DIR_ADDR   = './Output/'
NEURAL_NET_CONFIG = './NeuralNetwork/config.json'

def make_output_dirs(rootdir, clusters):
    # Clears ouput directory before starting:
    if os.path.isdir (rootdir):
        shutil.rmtree(rootdir)
    
    for cluster_name, cluster in clusters.items():
        for city in cluster.cities_dict.keys():
            os.makedirs(f'{rootdir}/cluster {cluster_name}/model {cluster_name}/city {city}')

def instantiate_ml_models_for_central_cities():
    neural_network_models = dict.fromkeys(clusters)
    
    for central_city in neural_network_models.keys():
        DATASET = clusters   [central_city].cities_dict[central_city]
        neural_network_models[central_city] = NeuralNetwork(NEURAL_NET_CONFIG, DATASET, THE_PLOTTER)
        print(f'\tCreated ML model {central_city}')
        
        tf.keras.backend.clear_session()
        
    return neural_network_models

def train_ml_models_for_central_cities():
    metrics_df_central_cities = None
    
    for neural_network_model_name, neural_network_model in neural_network_models.items():
        metrics_df_current_central_city, _ = neural_network_model.use_neural_network()

        if metrics_df_central_cities is None or metrics_df_central_cities.empty:
            metrics_df_central_cities = metrics_df_current_central_city
        else:
            metrics_df_central_cities = pd.concat([metrics_df_central_cities, metrics_df_current_central_city], ignore_index=True)
        
    return metrics_df_central_cities

print('PREPARATION: START')
THE_PLOTTER = Plotter()

clusters = InputDataLoader(INPUT_DIR_ADDR).get_cluster_memberships()
print('\tLoaded all datasets')

make_output_dirs(OUTPUT_DIR_ADDR, clusters)
print('\tMade output directories for all cities')
print('PREPARATION: END')

print('CREATION: START')
neural_network_models = instantiate_ml_models_for_central_cities()
print('CREATION: END')

print('TRAINING: START')
train_ml_models_for_central_cities()
print('TRAINING: END')