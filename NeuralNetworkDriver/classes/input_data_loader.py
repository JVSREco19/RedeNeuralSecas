import os

from NeuralNetwork.classes import Dataset

class InputDataLoader:
    def __init__(self, rootdir):
        self._clusters = dict.fromkeys(os.listdir(rootdir))
        self._fill_cluster_memberships(rootdir)
        self._load_all_datasets       ()
        self._print_cluster_info      ()
        
    def _fill_cluster_memberships(self, rootdir):
        for cluster_name, cluster_members in self._clusters.items():
             cluster_members_names = [city_name.removesuffix('.xlsx') for city_name in os.listdir(f'{rootdir}/{cluster_name}/')]
             self._clusters[cluster_name] = dict.fromkeys(cluster_members_names, {})
    
    def _load_all_datasets(self):
        for cluster_name, cluster_members in self._clusters.items():
            for city_name, city_data in cluster_members.items():
                self._clusters[cluster_name][city_name] = Dataset(city_name, cluster_name)
    
    def _print_cluster_info(self):
        for cluster_name, cluster_members in self._clusters.items():
            # Compute cluster-level statistics for informational purposes
            cluster_spei_min = None
            cluster_spei_max = None
            
            for city_data in cluster_members.values():
                if cluster_spei_min is None or city_data.spei.min() < cluster_spei_min:
                    cluster_spei_min = city_data.spei.min()
                if cluster_spei_max is None or city_data.spei.max() > cluster_spei_max:
                    cluster_spei_max = city_data.spei.max()
            
            print()
            print(f'Cluster {cluster_name}:')
            print(f'\tMin. SPEI: {cluster_spei_min:.4f}')
            print(f'\tMax. SPEI: {cluster_spei_max:.4f}')
            
            for city_name, city_data in cluster_members.items():
                print()
                print(f'\tCity {city_name}:')
                print(f'\tMin. SPEI: {min(city_data.spei):.4f}')
                print(f'\tMax. SPEI: {max(city_data.spei):.4f}')
    
    def get_cluster_memberships(self):
        return self._clusters