import os
import shutil

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

clusters = InputDataLoader(INPUT_DIR_ADDR).get_cluster_memberships()

make_output_dirs(OUTPUT_DIR_ADDR, clusters)

