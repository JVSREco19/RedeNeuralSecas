import os

from NeuralNetwork.classes import Dataset, Cluster

class InputDataLoader:
    def __init__(self, rootdir):
        self._clusters = dict.fromkeys(os.listdir(rootdir))
        self._fill_cluster_memberships(rootdir)
        self._load_all_datasets       ()
        self._instantiate_clusters    ()
        
    def _fill_cluster_memberships(self, rootdir):
        for cluster_name, cluster_members in self._clusters.items():
            self._clusters[cluster_name] = [city_name.removesuffix('.xlsx') for city_name in os.listdir(f'{rootdir}/{cluster_name}/')]
    
    def _load_all_datasets(self):
        for cluster_name, cluster_members in self._clusters.items():
            self._clusters[cluster_name] = [Dataset(city_name, cluster_name) for city_name in cluster_members]
    
    def _instantiate_clusters(self):
        for cluster_name, cluster_members in self._clusters.items():
            self._clusters[cluster_name] = Cluster(cluster_name, cluster_members)
    
    def get_cluster_memberships(self):
        return self._clusters