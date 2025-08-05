import tensorflow as tf
import pandas     as pd
import os
import shutil
from NeuralNetwork.classes import Dataset, NeuralNetwork, Plotter, PerformanceEvaluator

def define_cities_of_interest(rootdir):
    # Gets the names of the directories inside rootdir and defines them as the central cities:
    central_city_names   = os.listdir(rootdir)
    
    # Gets the names of the subdirectories inside each central city directory and defines them as the corresponding bordering cities:
    dict_cities_of_interest = {}
    for central_city_name in central_city_names:
        bordering_cities_filenames = os.listdir(f'{rootdir}/{central_city_name}/')
        
        bordering_cities_names= []
        for bordering_city_filename in bordering_cities_filenames:
            bordering_city_name = bordering_city_filename.rstrip('.xlsx')
            bordering_cities_names.append(bordering_city_name)
            
        # To avoid duplication of the central_city_name inside the bordering_cities_names:
        bordering_cities_names.remove(central_city_name)
            
        dict_cities_of_interest[central_city_name] = bordering_cities_names
        
    return dict_cities_of_interest

def create_empty_image_directory_tree(dict_cities_of_interest, rootdir):
    if os.path.isdir(rootdir):
        shutil.rmtree(rootdir)
    
    for central_city, list_of_bordering_cities in dict_cities_of_interest.items():
        os.makedirs(f'{rootdir}/cluster {central_city}/model {central_city}/city {central_city}')
        
        for bordering_city in list_of_bordering_cities:
            os.makedirs(f'{rootdir}/cluster {central_city}/model {central_city}/city {bordering_city}')

def load_all_datasets(dict_cities_of_interest):
    neural_network_datasets = {}
    
    for central_city, list_of_bordering_cities in dict_cities_of_interest.items():
        neural_network_datasets[central_city] = Dataset(central_city, central_city, INPUT_DATA_DIR, f'{central_city}/{central_city}.xlsx')
        
        for bordering_city in list_of_bordering_cities:
            neural_network_datasets[bordering_city] = Dataset(bordering_city, central_city, INPUT_DATA_DIR, f'{central_city}/{bordering_city}.xlsx')
    
    return neural_network_datasets

def instantiate_all_plotters(dict_cities_of_interest, neural_network_datasets):
    neural_network_plotters = {}

    for central_city, list_of_bordering_cities in dict_cities_of_interest.items():
        neural_network_plotters[central_city] = Plotter (neural_network_datasets[central_city])
        
        for bordering_city in list_of_bordering_cities:
            neural_network_plotters[bordering_city] = Plotter (neural_network_datasets[bordering_city])

    return neural_network_plotters

def create_ml_models_for_central_cities(dict_cities_of_interest, neural_network_datasets, neural_network_plotters, rootdir):
    neural_network_models   = {}
    
    for central_city in dict_cities_of_interest.keys():
        DATASET = neural_network_datasets[central_city]
        PLOTTER = neural_network_plotters[central_city]
        
        neural_network_models [central_city] = NeuralNetwork (NEURAL_NETWORK_CONFIG, DATASET, PLOTTER)
        print(f'\tCreated ML model {central_city}')
        
        tf.keras.backend.clear_session()
        
    return neural_network_models, neural_network_plotters, neural_network_datasets

def train_ml_models_for_central_cities():
    metrics_df_central_cities = None
    
    for neural_network_model_name, neural_network_model in neural_network_models.items():
        metrics_df_current_central_city, _ = neural_network_model.use_neural_network()

        if metrics_df_central_cities is None or metrics_df_central_cities.empty:
            metrics_df_central_cities = metrics_df_current_central_city
        else:
            metrics_df_central_cities = pd.concat([metrics_df_central_cities, metrics_df_current_central_city], ignore_index=True)
        
    return metrics_df_central_cities

def apply_ml_models_for_bordering_cities(dict_cities_of_interest, neural_network_models):
    metrics_df_bordering_cities = None
    
    for central_city, list_of_bordering_cities in dict_cities_of_interest.items():
        print(f'Model {central_city}:')
        MODEL = neural_network_models[central_city]
        
        for bordering_city in list_of_bordering_cities:
            # These will be input parameters for the neural network:
            print(f'\tCity {bordering_city}')
            DATASET = neural_network_datasets[bordering_city]
            PLOTTER = neural_network_plotters[bordering_city]
            
            _ , metrics_df_bordering_cities_current_model = MODEL.use_neural_network(dataset=DATASET, plotter=PLOTTER)

        # Run once for every central city, not for every bordering city:
        if metrics_df_bordering_cities is None:
            metrics_df_bordering_cities = metrics_df_bordering_cities_current_model
        else:
            metrics_df_bordering_cities = pd.concat([metrics_df_bordering_cities, metrics_df_bordering_cities_current_model], ignore_index=True)

    return metrics_df_bordering_cities
            
def save_ml_models_for_later_reuse(neural_network_models):
    if os.path.isdir('Models'):
        shutil.rmtree('Models')
    os.makedirs('Models')
    
    for name, model_object in neural_network_models.items():
        model_object.model.save        (f'Models/{name}.keras'     )
        model_object.model.save_weights(f'Models/{name}.weights.h5')


INPUT_DATA_DIR        = './Data/'
OUTPUT_IMAGE_DIR      = './Images/'
NEURAL_NETWORK_CONFIG = './NeuralNetwork/config.json'

print('PREPARATION: START')
dict_cities_of_interest = define_cities_of_interest(INPUT_DATA_DIR)
print('\tGot the names of the central cities and respective bordering ones')

create_empty_image_directory_tree(dict_cities_of_interest, OUTPUT_IMAGE_DIR)
print('\tMade output directories for all cities')

neural_network_datasets = load_all_datasets       (dict_cities_of_interest)
print('\tInstantiated all datasets')

neural_network_plotters = instantiate_all_plotters(dict_cities_of_interest, neural_network_datasets)
print('\tInstantiated all plotters')
print('PREPARATION: END')

print('CREATION: START')
(neural_network_models  ,
 neural_network_plotters,
 neural_network_datasets) = create_ml_models_for_central_cities(dict_cities_of_interest, neural_network_datasets, neural_network_plotters, INPUT_DATA_DIR)
print('CREATION: END')

print('TRAINING: START')
metrics_df_central_cities_only = train_ml_models_for_central_cities()
print('TRAINING: END')

print('APPLYING: START')
metrics_df_all_bordering_cities = apply_ml_models_for_bordering_cities(dict_cities_of_interest, neural_network_models)
print('APPLYING: END')

print('TERMINATION: START')
# Disabled, as these are not going to be used on Anderson's masters dissertation:
# any_plotter = list(neural_network_plotters.values())[0]
# any_plotter.plotMetricsPlots(metrics_df)

# Clustering isn't much important for OneToMany, as it is redundant with 'Municipio Treinado'. It is, however, very important for ManyToMany.
metrics_df_all_bordering_cities = metrics_df_all_bordering_cities.drop('Agrupamento', axis='columns')
metrics_df_central_cities_only  = metrics_df_central_cities_only .drop('Agrupamento', axis='columns')

metrics_df_all_bordering_cities.to_excel('metrics_bordering_cities.xlsx', index=False)
metrics_df_central_cities_only .to_excel('metrics_central_cities.xlsx'  , index=False)

save_ml_models_for_later_reuse(neural_network_models)
print('TERMINATION: END')