import pandas as pd
import numpy  as np
from sklearn.model_selection import train_test_split

class Dataset:
    
    DATA_PORTION_TYPES = ['80%', '20%'] # '100%' is made out of 80% + 20% through 'concatenate'
    
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
        
        #         IN               ,           OUT               :
        spei_provided_inputs       , spei_expected_outputs       =  self._create_input_output_pairs(  spei_dict, configs_dict)
        months_for_provided_inputs , months_for_expected_outputs =  self._create_input_output_pairs(months_dict, configs_dict)
        
        ###100% DATA PORTIONS##################################################
        spei_provided_inputs        ['100%'] = np.concatenate( (spei_provided_inputs        ['80%'] ,
                                                                spei_provided_inputs        ['20%']), axis=0)
        spei_expected_outputs       ['100%'] = np.concatenate( (spei_expected_outputs       ['80%'] ,
                                                                spei_expected_outputs       ['20%']), axis=0)
        
        months_for_provided_inputs  ['100%'] = np.concatenate( (months_for_provided_inputs  ['80%'] ,
                                                                months_for_provided_inputs  ['20%']), axis=0)
        months_for_expected_outputs ['100%'] = np.concatenate( (months_for_expected_outputs ['80%'] ,
                                                                months_for_expected_outputs ['20%']), axis=0)
        #######################################################################
        return (                  spei_dict,                months_dict  ,
                      spei_provided_inputs , spei_expected_outputs       ,
                months_for_provided_inputs , months_for_expected_outputs )
    
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
            spei_dict['80%']  = np.zeros_like(spei_dict['80%'])
            spei_dict['20%']  = np.zeros_like(spei_dict['20%'])
            spei_dict['100%'] = np.zeros_like(spei_dict['100%'])
        else:
            spei_dict['80%']  = (spei_dict['80%']  - self.spei_min) / spei_delta
            spei_dict['20%']  = (spei_dict['20%']  - self.spei_min) / spei_delta
            spei_dict['100%'] = (spei_dict['100%'] - self.spei_min) / spei_delta
        
        # Store normalized full dataset for backward compatibility
        self.spei_normalized = spei_dict['100%']
                                                                    
        return spei_dict, months_dict
    
    def _create_input_output_pairs(self, data_dict, configs_dict):
        window_gap  = configs_dict['total_points']
        dense_units = configs_dict['dense_units' ]
        
        input_dict  = dict.fromkeys(Dataset.DATA_PORTION_TYPES)
        output_dict = dict.fromkeys(Dataset.DATA_PORTION_TYPES)
        
        for data_portion_type in Dataset.DATA_PORTION_TYPES:
            # Data → sliding windows (with overlaps):
            windows = np.lib.stride_tricks.sliding_window_view(data_dict[data_portion_type], window_gap)
            
            # -overlaps by selecting only every 'window_gap'-th window:
            windows = windows[::window_gap]
            
            # Last 'dense_units' elements from each window → output;
            # Remaining elements in each window            → input :
            output_dict[data_portion_type] = windows[ : , -dense_units :              ]
            input_dict [data_portion_type] = windows[ : ,              : -dense_units ]
            
            # +new dimension at the end of the array:
            input_dict[data_portion_type] = input_dict[data_portion_type][..., np.newaxis]
        
        return input_dict, output_dict