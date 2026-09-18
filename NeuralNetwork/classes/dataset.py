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
        # self.spei_normalized   = None
        # self.spei_min          = None
        # self.spei_max          = None

    def get_months(self):
        return self.months
    
    def get_spei(self):
        return self.spei
    
    # def get_spei_normalized(self):
        # return self.spei_normalized
    
    def _slice_printer(self):
        for win_num in range(0, (len(self.label_col)//self.total_win_size) ):
            
            # Funciona:
            print(f"INPUT : {self.label_col[slice(win_num                   , win_num + self.input_width)]}")
            
            # Não funciona (imprime "OUTPUT: []"):
            print(f"OUTPUT: {self.label_col[slice(win_num + self.input_width, win_num + self.shift      )]}")
            
    
    def _dataset_windower(self, input_width, label_width):
        self.input_width    = input_width
        self.label_width    = label_width
        # I'm not interested in leaving gaps/skips between input and labels:
        self.shift          = label_width
        self.total_win_size = self.input_width + self.shift
        
        self.label_col      = self.get_spei()
        self.input_slice    = slice(0, self.input_width)
        
        self._slice_printer()
        
        print()
        
    def get_tumbling_windows(self, input_width, label_width):
        self._dataset_windower(input_width, label_width)
    
    # def _window_maker(self, data_dict, configs_dict):
        
    #     data = {key: {} for key in Dataset.DATA_TECHNIQUE_TYPES}
        
    #     for technique in Dataset.DATA_TECHNIQUE_TYPES:
            
    #         window_len   = configs_dict[f'{technique}_window_len'  ]
    #         window_step  = configs_dict[f'{technique}_window_step' ]
    #         lookback_len = configs_dict[f'{technique}_lookback_len']
    #         horizon_len  = configs_dict[f'{technique}_horizon_len' ]
            
    #         input_data  = dict.fromkeys(Dataset.DATA_PORTION_TYPES)
    #         output_data = dict.fromkeys(Dataset.DATA_PORTION_TYPES)
            
    #         for data_portion_type in Dataset.DATA_PORTION_TYPES:
    #             # Data → sliding windows (with overlaps):
    #             windows = sliding_windower(x    = data_dict[data_portion_type],
    #                                        window_shape = window_len          )
                
    #             # reduces overlaps
    #             windows = windows[::window_step]
                
    #             input_data [data_portion_type] = windows[ : ,              : lookback_len]
    #             output_data[data_portion_type] = windows[ : , -horizon_len :             ]
                
    #             # +new dimension at the end of the array:
    #             input_data[data_portion_type] = input_data[data_portion_type][..., np.newaxis]
            
    #         data[technique].update({'input' :  input_data})
    #         data[technique].update({'output': output_data})
            
    #     return data