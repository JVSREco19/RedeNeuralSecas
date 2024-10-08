#Deve-se passar o caminho para o xlsx da região para qual o modelo deve ser TREINADO
from NeuralNetwork.NeuralNetwork import ApplyTraining, FitNeuralNetwork,PrintMetricsList

import pandas as pd
import numpy as np
import json
import os

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
    for central_city, list_of_bordering_cities in dict_cities_of_interest.items():
        os.makedirs(f'{rootdir}/{central_city}/{central_city}')
        
        for bordering_city in list_of_bordering_cities:
            os.makedirs(f'{rootdir}/{central_city}/{bordering_city}')
    
def create_neural_network_models_for_central_cities(dict_cities_of_interest, rootdir):
    neural_network_models = {}
    
    for central_city in dict_cities_of_interest.keys():
        model, metricsCompendium = FitNeuralNetwork(f'{rootdir}/{central_city}/{central_city}.xlsx', central_city, central_city, SHOW_IMAGES)
        neural_network_models[central_city] = model
    return neural_network_models, metricsCompendium

def apply_neural_network_models_for_bordering_cities(dict_cities_of_interest, neural_network_models, rootdir):
    for central_city, list_of_bordering_cities in dict_cities_of_interest.items():
        for bordering_city in list_of_bordering_cities:
            metricsCompendium = ApplyTraining(f'{rootdir}/{central_city}/{bordering_city}.xlsx', central_city, bordering_city, neural_network_models[central_city], SHOW_IMAGES, bordering_city)
    return metricsCompendium


SHOW_IMAGES = False

dict_cities_of_interest = define_cities_of_interest('./Data')

create_empty_image_directory_tree(dict_cities_of_interest, './Images')

print('TRAINING: START')
neural_network_models, metricsCompendium = create_neural_network_models_for_central_cities(dict_cities_of_interest, './Data')
print('TRAINING: END')

print('APPLYING: START')
metricsCompendium = apply_neural_network_models_for_bordering_cities(dict_cities_of_interest, neural_network_models, './Data')
print('APPLYING: END')

PrintMetricsList(metricsCompendium)
