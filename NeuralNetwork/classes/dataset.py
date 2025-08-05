import pandas as pd
import numpy  as np
from sklearn.model_selection import train_test_split

class Dataset:
    
    DATA_PORTION_TYPES = ['80%', '20%'] # '100%' is made out of 80% + 20% through 'concatenate'
    
    def __init__(self, city_name, city_cluster_name, root_dir, xlsx):
        self.city_name         = city_name
        self.city_cluster_name = city_cluster_name
        self.df                = pd.read_excel(root_dir + xlsx, index_col=0)
        self.df.rename(columns = {'Series 1': 'SPEI Real'}, inplace=True)
        self.months            = self.df.index.to_numpy()
        self.spei              = self.df['SPEI Real'].to_numpy()
        self.spei_normalized   = ( (self.spei       - self.spei.min()) /
                                   (self.spei.max() - self.spei.min()) )

    def get_months(self):
        return self.months
    
    def get_spei(self):
        return self.spei
    
    def get_spei_normalized(self):
        return self.spei_normalized
    
    def format_data_for_model(self, configs_dict):
        #(SPEI/months)_dict.keys() = ['80%', '20%']
        spei_dict                  , months_dict                = self._train_test_split(configs_dict['parcelDataTrain'])
        
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
    
    def _train_test_split(self, train_size):
        
        spei_dict   = dict.fromkeys(Dataset.DATA_PORTION_TYPES)
        months_dict = dict.fromkeys(Dataset.DATA_PORTION_TYPES)
        
        spei_dict  ['100%'] = self.get_spei_normalized()
        months_dict['100%'] = self.get_months         ()
        
        (  spei_dict['80%'],   spei_dict['20%'],
         months_dict['80%'], months_dict['20%']) = train_test_split(spei_dict  ['100%']     ,
                                                                    months_dict['100%']     ,
                                                                    train_size = train_size ,
                                                                    shuffle    = False      )
                                                                    
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