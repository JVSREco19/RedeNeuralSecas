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
    
    for cluster_name, cities_dict in clusters.items():
        for city in cities_dict.keys():
            os.makedirs(f'{rootdir}/cluster {cluster_name}/model {cluster_name}/city {city}')

def instantiate_ml_models_for_central_cities():
    neural_network_models = dict.fromkeys(clusters)
    
    for central_city in neural_network_models.keys():
        DATASET = clusters[central_city][central_city]
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

def apply_ml_models_for_bordering_cities(clusters, neural_network_models):
    
    metrics_df_bordering_cities = None
    
    for cluster_name, cities_dict in clusters.items():
        print(f'Model {cluster_name}:')
        MODEL = neural_network_models[cluster_name]
        
        bordering_cities = list(cities_dict.keys())
        bordering_cities.remove(cluster_name)
        
        for city in bordering_cities:
            print(f'\tCity {city}')
            DATASET = clusters[cluster_name][city]
            
            _ , metrics_df_bordering_cities_current_model = MODEL.use_neural_network(dataset=DATASET)
    
        # Run once for every central city, not for every bordering city:
        if metrics_df_bordering_cities is None:
            metrics_df_bordering_cities = metrics_df_bordering_cities_current_model
        else:
            metrics_df_bordering_cities = pd.concat([metrics_df_bordering_cities, metrics_df_bordering_cities_current_model], ignore_index=True)
    
    return metrics_df_bordering_cities

def save_ml_models_for_later_reuse(neural_network_models):
    if os.path.isdir(f'{OUTPUT_DIR_ADDR}/Models'):
        shutil.rmtree(f'{OUTPUT_DIR_ADDR}/Models')
    os.makedirs(f'{OUTPUT_DIR_ADDR}/Models')
    
    for name, model_object in neural_network_models.items():
        model_object.model.save        (f'{OUTPUT_DIR_ADDR}/Models/{name}.keras'     )
        model_object.model.save_weights(f'{OUTPUT_DIR_ADDR}/Models/{name}.weights.h5')

def save_results(metrics_df_all_bordering_cities, metrics_df_central_cities_only, neural_network_models):
    # Sort by cluster name first, then by city name
    metrics_df_all_bordering_cities = metrics_df_all_bordering_cities.sort_values(
        by=['Agrupamento', 'Municipio Previsto'], ignore_index=True)
    metrics_df_central_cities_only = metrics_df_central_cities_only.sort_values(
        by=['Agrupamento', 'Municipio Previsto'], ignore_index=True)
    
    metrics_df_all_bordering_cities = metrics_df_all_bordering_cities.drop('Agrupamento', axis='columns')
    metrics_df_central_cities_only  = metrics_df_central_cities_only .drop('Agrupamento', axis='columns')

    metrics_df_all_bordering_cities.to_excel(f'{OUTPUT_DIR_ADDR}/metrics_bordering_cities.xlsx', index=False)
    metrics_df_central_cities_only .to_excel(f'{OUTPUT_DIR_ADDR}/metrics_central_cities.xlsx'  , index=False)

    save_ml_models_for_later_reuse(neural_network_models)

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
metrics_df_central_cities_only = train_ml_models_for_central_cities()
print('TRAINING: END')

print('APPLYING: START')
metrics_df_all_bordering_cities = apply_ml_models_for_bordering_cities(clusters, neural_network_models)
print('APPLYING: END')

print('TERMINATION: START')
save_results(metrics_df_all_bordering_cities, metrics_df_central_cities_only, neural_network_models)
print('TERMINATION: END')