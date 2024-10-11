#Deve-se passar o caminho para o xlsx da região para qual o modelo deve ser TREINADO
from NeuralNetwork.NeuralNetwork import ApplyTraining, FitNeuralNetwork,PrintMetricsList

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
            
        dict_cities_of_interest[central_city_name] = bordering_cities_names
        
    return dict_cities_of_interest

def create_empty_image_directory_tree(dict_city_clusters, rootdir):
    for city_cluster_name, list_of_cities in dict_city_clusters.items():
#        print(f'City cluster name: {city_cluster_name}. Cointains {len(list_of_cities)} cities.')
        
        for city_for_training in list_of_cities:
            os.makedirs(f'{rootdir}/cluster {city_cluster_name}/model {city_for_training}')
            
            for city_for_predicting in list_of_cities:
                os.makedirs(f'{rootdir}/cluster {city_cluster_name}/model {city_for_training}/city {city_for_predicting}')
            
SHOW_IMAGES = False

dict_city_clusters = define_cities_of_interest('./Data')

create_empty_image_directory_tree(dict_city_clusters, './Images')

for city_cluster_name, list_of_cities in dict_city_clusters.items():
    print(f'City cluster name: {city_cluster_name}. Cointains {len(list_of_cities)} cities.')

    for city_for_training in list_of_cities:
        print(f'FitNeuralNetwork : {city_for_training}')
        model, metricsCompendium = FitNeuralNetwork(f'./Data/{city_cluster_name}/{city_for_training}.xlsx', city_cluster_name, city_for_training, city_for_training, SHOW_IMAGES)
        
        list_of_other_cities = list_of_cities.copy()
        list_of_other_cities.remove(city_for_training)
        
        for city_for_prediction in list_of_other_cities:
            print(f'\tApplyTraining: {city_for_prediction}')
            metricsCompendium = ApplyTraining(f'./Data/{city_cluster_name}/{city_for_prediction}.xlsx', city_cluster_name, city_for_training, city_for_prediction, model, SHOW_IMAGES, city_for_prediction)

PrintMetricsList(metricsCompendium)