import tensorflow as tf
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

def create_neural_network_models_for_central_cities(dict_cities_of_interest, neural_network_datasets, neural_network_plotters, rootdir):
    neural_network_models   = {}
    
    for central_city in dict_cities_of_interest.keys():
        DATASET = neural_network_datasets[central_city]
        PLOTTER = neural_network_plotters[central_city]
        
        neural_network_models [central_city] = NeuralNetwork (NEURAL_NETWORK_CONFIG, DATASET, PLOTTER)
        
        tf.keras.backend.clear_session()
        
    return neural_network_models, neural_network_plotters, neural_network_datasets

def train_neural_network_models_for_central_cities():
    metrics_df_central_cities = {}
    
    for neural_network_model_name, neural_network_model in neural_network_models.items():
        metrics_df = neural_network_model.use_neural_network()
        metrics_df_central_cities[neural_network_model_name] = metrics_df
        
    return metrics_df_central_cities

def apply_neural_network_models_for_bordering_cities(dict_cities_of_interest, neural_network_models):
    #(dict_cities_of_interest, neural_network_models, rootdir):
    
    for central_city, list_of_bordering_cities in dict_cities_of_interest.items():
        print(f'Model {central_city}:')
        for bordering_city in list_of_bordering_cities:
            print(f'\tCity {bordering_city}')
            DATASET = neural_network_plotters[bordering_city]
            PLOTTER = neural_network_datasets[bordering_city]
            
            metrics_df = neural_network_models[central_city].use_neural_network(dataset=DATASET, plotter=PLOTTER)
            # One metrics_df for each bordering city. This needs to be properly addressed.
# apply_neural_network_models_for_bordering_cities(dict_cities_of_interest, neural_network_models)


INPUT_DATA_DIR        = './Data/'
OUTPUT_IMAGE_DIR      = './Images/'
NEURAL_NETWORK_CONFIG = './NeuralNetwork/config.json'

dict_cities_of_interest = define_cities_of_interest(INPUT_DATA_DIR)

create_empty_image_directory_tree(dict_cities_of_interest, OUTPUT_IMAGE_DIR)

print('PREPARATION: START')
neural_network_datasets = load_all_datasets       (dict_cities_of_interest)
print('\tLoaded all datasets')

# TO-DO:
neural_network_plotters = instantiate_all_plotters(dict_cities_of_interest, neural_network_datasets)
print('\tCreated all plotters')
print('PREPARATION: END')

print('CREATION: START')
(neural_network_models  ,
 neural_network_plotters,
 neural_network_datasets) = create_neural_network_models_for_central_cities(dict_cities_of_interest, neural_network_datasets, neural_network_plotters, INPUT_DATA_DIR)
print('CREATION: END')

print('TRAINING: START')
metrics_df_central_cities = train_neural_network_models_for_central_cities()
print('TRAINING: END')

print('APPLYING: START')
# metrics_df = apply_neural_network_models_for_bordering_cities(dict_cities_of_interest, neural_network_models, './Data')
print('APPLYING: END')

### NEW CODE: ###

# [...]

# metrics_df = rio_pardo_de_mg_model.use_neural_network ()
# rio_pardo_de_mg_plotter.plotMetricsPlots              (metrics_df)

# montezuma_dataset         = Dataset ('Montezuma', 'Rio Pardo de Minas', './Data/', 'MONTEZUMA.xlsx')
# montezuma_plotter         = Plotter (montezuma_dataset)

# metrics_df = rio_pardo_de_mg_model.use_neural_network (dataset=montezuma_dataset, plotter=montezuma_plotter)
# montezuma_plotter.plotMetricsPlots                 (metrics_df)
#######

### OLD CODE: ###

# [...]

# metrics_df.to_excel('metricas_modelo.xlsx', index=False)

# metrics_df = metrics_df.drop('Agrupamento', axis='columns') # Clustering isn't much important for OneToMany, as it is redundant with 'Municipio Treinado'. It is, however, very important for ManyToMany.

# drawMetricsBoxPlots   (metrics_df, SHOW_IMAGES)
# drawMetricsBarPlots   (metrics_df, SHOW_IMAGES)
# drawMetricsHistograms (metrics_df, SHOW_IMAGES)
# drawMetricsRadarPlots (metrics_df, SHOW_IMAGES)
#######