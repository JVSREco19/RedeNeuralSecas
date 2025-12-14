class Cluster:
    
    def __init__(self, name, cities_list):
        self.name          = str(name).upper()
        self.root_dir      = f'./Data/{self.name}'
        self.cities_list   = cities_list;
        
        self.spei_min      = self._get_spei_min()
        self.spei_max      = self._get_spei_max()
        self.spei_min_norm = None
        self.spei_max_norm = None
        self._normalize_cities_SPEI()
        
        self._print_cluster_info()
    
    def _get_spei_min(self):
        cluster_spei_min = None
        
        for city in self.cities_list:
            if cluster_spei_min is None or city.spei.min() < cluster_spei_min:
                cluster_spei_min = city.spei.min()
        
        return cluster_spei_min
    
    def _get_spei_max(self):
        cluster_spei_max = None
        
        for city in self.cities_list:
            if cluster_spei_max is None or city.spei.max() > cluster_spei_max:
                cluster_spei_max = city.spei.max()
        
        return cluster_spei_max
    
    def _normalize_cities_SPEI(self):
        for city in self.cities_list:
            city.spei_normalized = ( (city.spei       - self.spei_min) /
                                     (self.spei_max   - self.spei_min) )
            
            if self.spei_min_norm is None or min(city.spei_normalized) < self.spei_min_norm:
                self.spei_min_norm = min(city.spei_normalized)
                
            if self.spei_max_norm is None or max(city.spei_normalized) > self.spei_max_norm:
                self.spei_max_norm = max(city.spei_normalized)
    
    def _print_cluster_info(self):
        print()
        print(f'Cluster {self.name}:')
        print(f'\tMin. SPEI: {self.spei_min_norm:.4f} = {self.spei_min:.4f}')
        print(f'\tMax. SPEI: {self.spei_max_norm:.4f} = {self.spei_max:.4f}')
        
        for city in self.cities_list:
            print()
            print(f'\tCity {city.city_name}:')
            print(f'\tMin. SPEI: {min(city.spei_normalized):.4f} = {min(city.spei):.4f}')
            print(f'\tMax. SPEI: {max(city.spei_normalized):.4f} = {max(city.spei):.4f}')