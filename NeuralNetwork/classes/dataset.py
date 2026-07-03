import pandas as pd
import numpy  as np
from numpy.lib.stride_tricks import sliding_window_view as sliding_windower
from sklearn.model_selection import train_test_split

class Dataset:
    
    DATA_PORTION_TYPES   = ['80%', '20%'] # '100%' is made out of 80% + 20% through 'concatenate'
    DATA_TECHNIQUE_TYPES = ['tumbling', 'sliding']
    
    def __init__(self, city_name, city_cluster_name):
        self.city_name         = str(city_name        ).upper()
        self.city_cluster_name = str(city_cluster_name).upper()
        self.df                = pd.read_excel(f'./Data/{self.city_cluster_name}/{self.city_name}.xlsx', index_col=0)
        self.df.rename(columns = {'Series 1': 'SPEI Real'}, inplace=True)
        self.months            = self.df.index.to_numpy()
        self.spei              = self.df['SPEI Real'].to_numpy()
        self.spei_normalized   = None
        self.spei_min          = None
        self.spei_max          = None

    def get_months(self):
        return self.months
    
    def get_spei(self):
        return self.spei
    
    def get_spei_normalized(self):
        return self.spei_normalized
    
    def format_data_for_model(self, configs_dict, norm_min=None, norm_max=None):
        #(SPEI/months)_dict.keys() = ['80%', '20%']
        spei_dict                  , months_dict                = self._train_test_split(configs_dict['parcelDataTrain'], norm_min, norm_max)
        
        spei_data   =  self._window_maker(  spei_dict, configs_dict)
        months_data =  self._window_maker(months_dict, configs_dict)
        
        ###100% DATA PORTIONS TUMBLING#########################################
        spei_data  ['tumbling']['input' ]['100%'] = np.concatenate( (spei_data  ['tumbling']['input' ]['80%'] ,
                                                                     spei_data  ['tumbling']['input' ]['20%']), axis=0)
        spei_data  ['tumbling']['output']['100%'] = np.concatenate( (spei_data  ['tumbling']['output']['80%'] ,
                                                                     spei_data  ['tumbling']['output']['20%']), axis=0)
        
        months_data['tumbling']['input' ]['100%'] = np.concatenate( (months_data['tumbling']['input' ]['80%'] ,
                                                                     months_data['tumbling']['input' ]['20%']), axis=0)
        months_data['tumbling']['output']['100%'] = np.concatenate( (months_data['tumbling']['output']['80%'],
                                                                     months_data['tumbling']['output']['20%']), axis=0)
        ###100% DATA PORTIONS SLIDING##########################################
        spei_data  ['sliding' ]['input' ]['100%'] = np.concatenate( (spei_data  ['sliding' ]['input' ]['80%'] ,
                                                                     spei_data  ['sliding' ]['input' ]['20%']), axis=0)
        spei_data  ['sliding' ]['output']['100%'] = np.concatenate( (spei_data  ['sliding' ]['output']['80%'] ,
                                                                     spei_data  ['sliding' ]['output']['20%']), axis=0)
        
        months_data['sliding' ]['input' ]['100%'] = np.concatenate( (months_data['sliding' ]['input' ]['80%'] ,
                                                                     months_data['sliding' ]['input' ]['20%']), axis=0)
        months_data['sliding' ]['output']['100%'] = np.concatenate( (months_data['sliding' ]['output']['80%'],
                                                                     months_data['sliding' ]['output']['20%']), axis=0)      
        #######################################################################
        return (spei_dict, months_dict,
                spei_data, months_data)
    
    def _train_test_split(self, train_size, norm_min=None, norm_max=None):
        
        spei_dict   = dict.fromkeys(Dataset.DATA_PORTION_TYPES)
        months_dict = dict.fromkeys(Dataset.DATA_PORTION_TYPES)
        
        # Split data BEFORE normalization
        spei_dict  ['100%'] = self.get_spei()
        months_dict['100%'] = self.get_months()
        
        (  spei_dict['80%'],   spei_dict['20%'],
         months_dict['80%'], months_dict['20%']) = train_test_split(spei_dict  ['100%']     ,
                                                                    months_dict['100%']     ,
                                                                    train_size = train_size ,
                                                                    shuffle    = False      )
        
        # Normalize using provided parameters or compute from training set
        if norm_min is not None and norm_max is not None:
            # Use provided normalization parameters (for bordering cities)
            self.spei_min = norm_min
            self.spei_max = norm_max
        else:
            # Compute normalization parameters from training set only (for central cities)
            self.spei_min = spei_dict['80%'].min()
            self.spei_max = spei_dict['80%'].max()
        
        # Apply normalization to all portions
        # Check for zero variance to avoid division by zero
        spei_delta = self.spei_max - self.spei_min
        if np.isclose(spei_delta, 0):
            # If all values are the same, normalized values should be 0
            spei_dict[ '80%'] = np.zeros_like(spei_dict[ '80%'])
            spei_dict[ '20%'] = np.zeros_like(spei_dict[ '20%'])
            spei_dict['100%'] = np.zeros_like(spei_dict['100%'])
        else:
            spei_dict[ '80%'] = (spei_dict[ '80%'] - self.spei_min) / spei_delta
            spei_dict[ '20%'] = (spei_dict[ '20%'] - self.spei_min) / spei_delta
            spei_dict['100%'] = (spei_dict['100%'] - self.spei_min) / spei_delta
        
        # Store normalized full dataset for backward compatibility
        self.spei_normalized = spei_dict['100%']
                                                                    
        return spei_dict, months_dict
    
    def _window_maker(self, data_dict, configs_dict):
        
        data = {key: {} for key in Dataset.DATA_TECHNIQUE_TYPES}
        
        for technique in Dataset.DATA_TECHNIQUE_TYPES:
            
            window_len   = configs_dict[f'{technique}_window_len'  ]
            window_step  = configs_dict[f'{technique}_window_step' ]
            lookback_len = configs_dict[f'{technique}_lookback_len']
            horizon_len  = configs_dict[f'{technique}_horizon_len' ]
            
            input_data  = dict.fromkeys(Dataset.DATA_PORTION_TYPES)
            output_data = dict.fromkeys(Dataset.DATA_PORTION_TYPES)
            
            for data_portion_type in Dataset.DATA_PORTION_TYPES:
                # Data → sliding windows (with overlaps):
                windows = sliding_windower(x    = data_dict[data_portion_type],
                                           window_shape = window_len          )
                
                # reduces overlaps
                windows = windows[::window_step]
                
                input_data [data_portion_type] = windows[ : ,              : lookback_len]
                output_data[data_portion_type] = windows[ : , -horizon_len :             ]
                
                # +new dimension at the end of the array:
                input_data[data_portion_type] = input_data[data_portion_type][..., np.newaxis]
            
            data[technique].update({'input' :  input_data})
            data[technique].update({'output': output_data})
            
        return data