import pandas as pd
import numpy  as np
from sklearn.model_selection import train_test_split

class Dataset:
    
    DATA_TYPES_LIST = ['Train', 'Test']
    
    def __init__(self, city_name, city_cluster_name, root_dir, xlsx):
        self.city_name         = city_name
        self.city_cluster_name = city_cluster_name
        self.df                = pd.read_excel(root_dir + xlsx)
        self.df.rename(columns = {'Series 1': 'SPEI Real'}, inplace=True)

    def get_months(self):
        return self.df.index.to_numpy()
    
    def get_spei(self):
        return self.df['SPEI Real'].to_numpy()
    
    def get_spei_normalized(self):
        spei = self.get_spei()
        return ( (spei - spei.min()) / (spei.max() - spei.min()) )
    
    def format_data_for_model(self, configs_dict):
        #(SPEI/months)_dict.keys() = ['Train', 'Test']
        spei_dict               , months_dict             = self._train_test_split(configs_dict['parcelDataTrain'])
        
        #         IN            ,           OUT           :
        dataForPrediction_dict  , dataTrueValues_dict     =  self._create_input_output(  spei_dict, configs_dict)
        monthsForPrediction_dict, monthsForPredicted_dict =  self._create_input_output(months_dict, configs_dict)
        
        return (               spei_dict,             months_dict,
                  dataForPrediction_dict,     dataTrueValues_dict,
                monthsForPrediction_dict, monthsForPredicted_dict)
    
    def _train_test_split(self, train_size):
        
        spei_dict   = dict.fromkeys(Dataset.DATA_TYPES_LIST)
        months_dict = dict.fromkeys(Dataset.DATA_TYPES_LIST)
        
        (  spei_dict['Train'],   spei_dict['Test'],
         months_dict['Train'], months_dict['Test']) = train_test_split(self.get_spei_normalized(), self.get_months(), train_size=train_size, shuffle=False)
        
        return spei_dict, months_dict
    
    def _create_input_output(self, data_dict, configs_dict):
        window_gap  = configs_dict['total_points']
        dense_units = configs_dict['dense_units' ]
        
        input_dict  = dict.fromkeys(Dataset.DATA_TYPES_LIST)
        output_dict = dict.fromkeys(Dataset.DATA_TYPES_LIST)
        
        for train_or_test in Dataset.DATA_TYPES_LIST:
            # Data → sliding windows (with overlaps):
            windows = np.lib.stride_tricks.sliding_window_view(data_dict[train_or_test], window_gap)
            
            # -overlaps by selecting only every 'window_gap'-th window:
            windows = windows[::window_gap]
            
            # Last 'dense_units' elements from each window → output;
            # Remaining elements in each window            → input :
            output_dict[train_or_test] = windows[ : , -dense_units :              ]
            input_dict [train_or_test] = windows[ : ,              : -dense_units ]
            
            # +new dimension at the end of the array:
            input_dict[train_or_test] = input_dict[train_or_test][..., np.newaxis]
        
        return input_dict, output_dict