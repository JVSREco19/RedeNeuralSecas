class Cluster:
    
    def __init__(self, name, cities_dict):
        self.name          = str(name).upper()
        self.root_dir      = f'./Data/{self.name}'
        self.cities_dict   = cities_dict;
        
        self.spei_min      = self._get_spei_min()
        self.spei_max      = self._get_spei_max()
        self.spei_min_norm = None
        self.spei_max_norm = None
        self._normalize_cities_SPEI()
        
        self._print_cluster_info()
    
    def _get_spei_min(self):
        cluster_spei_min = None
        
        for city_data in self.cities_dict.values():
            if cluster_spei_min is None or city_data.spei.min() < cluster_spei_min:
                cluster_spei_min = city_data.spei.min()
        
        return cluster_spei_min
    
    def _get_spei_max(self):
        cluster_spei_max = None
        
        for city_data in self.cities_dict.values():
            if cluster_spei_max is None or city_data.spei.max() > cluster_spei_max:
                cluster_spei_max = city_data.spei.max()
        
        return cluster_spei_max
    
    def _normalize_cities_SPEI(self):
        for city_data in self.cities_dict.values():
            city_data.spei_normalized = ( (city_data.spei       - self.spei_min) /
                                     (self.spei_max   - self.spei_min) )
            
            if self.spei_min_norm is None or min(city_data.spei_normalized) < self.spei_min_norm:
                self.spei_min_norm = min(city_data.spei_normalized)
                
            if self.spei_max_norm is None or max(city_data.spei_normalized) > self.spei_max_norm:
                self.spei_max_norm = max(city_data.spei_normalized)
    
    def _print_cluster_info(self):
        print()
        print(f'Cluster {self.name}:')
        print(f'\tMin. SPEI: {self.spei_min_norm:.4f} = {self.spei_min:.4f}')
        print(f'\tMax. SPEI: {self.spei_max_norm:.4f} = {self.spei_max:.4f}')
        
        for city_name, city_data in self.cities_dict.items():
            print()
            print(f'\tCity {city_name}:')
            print(f'\tMin. SPEI: {min(city_data.spei_normalized):.4f} = {min(city_data.spei):.4f}')
            print(f'\tMax. SPEI: {max(city_data.spei_normalized):.4f} = {max(city_data.spei):.4f}')