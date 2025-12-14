import os
import shutil

# from NeuralNetwork      .classes import Cluster#, Dataset, NeuralNetwork, Plotter, PerformanceEvaluator
from NeuralNetworkDriver.classes import InputDataLoader

INPUT_DIR_ADDR        = './Data/'
OUTPUT_DIR_ADDR       = './Output/'

def make_output_dirs(rootdir, clusters):
    # Clears ouput directory before starting:
    if os.path.isdir (rootdir):
        shutil.rmtree(rootdir)
    
    for cluster_name, cluster in clusters.items():
        for city in cluster.cities_list:
            os.makedirs(f'{rootdir}/cluster {cluster_name}/model {cluster_name}/city {city.city_name}')

clusters = InputDataLoader(INPUT_DIR_ADDR).get_cluster_memberships()

make_output_dirs(OUTPUT_DIR_ADDR, clusters)

