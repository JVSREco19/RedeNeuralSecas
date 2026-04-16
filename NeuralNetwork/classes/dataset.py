import pandas as pd
import numpy  as np
from numpy.lib.stride_tricks import sliding_window_view as sliding_windower
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
        self._validate_window_configs(configs_dict)
        #(SPEI/months)_dict.keys() = ['80%', '20%']
        spei_dict                  , months_dict                = self._train_test_split(configs_dict['parcelDataTrain'], norm_min, norm_max)
        
        #              IN                   ,                OUT                  :
        (spei_provided_inputs_tumbling      , spei_expected_outputs_tumbling      ,
         spei_provided_inputs_sliding       , spei_expected_outputs_sliding       ) =  self._create_input_output_pairs(  spei_dict, configs_dict, "spei")
        (months_for_provided_inputs_tumbling, months_for_expected_outputs_tumbling,
         months_for_provided_inputs_sliding , months_for_expected_outputs_sliding ) =  self._create_input_output_pairs(months_dict, configs_dict, "months")
        
        ###100% DATA PORTIONS TUMBLING#########################################
        spei_provided_inputs_tumbling        ['100%'] = np.concatenate( (spei_provided_inputs_tumbling        ['80%'] ,
                                                                         spei_provided_inputs_tumbling        ['20%']), axis=0)
        spei_expected_outputs_tumbling       ['100%'] = np.concatenate( (spei_expected_outputs_tumbling       ['80%'] ,
                                                                         spei_expected_outputs_tumbling       ['20%']), axis=0)
        
        months_for_provided_inputs_tumbling  ['100%'] = np.concatenate( (months_for_provided_inputs_tumbling  ['80%'] ,
                                                                         months_for_provided_inputs_tumbling  ['20%']), axis=0)
        months_for_expected_outputs_tumbling ['100%'] = np.concatenate( (months_for_expected_outputs_tumbling ['80%'] ,
                                                                         months_for_expected_outputs_tumbling ['20%']), axis=0)
        ###100% DATA PORTIONS SLIDING##########################################
        spei_provided_inputs_sliding        ['100%'] = np.concatenate( (spei_provided_inputs_sliding          ['80%'] ,
                                                                        spei_provided_inputs_sliding          ['20%']), axis=0)
        spei_expected_outputs_sliding       ['100%'] = np.concatenate( (spei_expected_outputs_sliding         ['80%'] ,
                                                                        spei_expected_outputs_sliding         ['20%']), axis=0)
        
        months_for_provided_inputs_sliding  ['100%'] = np.concatenate( (months_for_provided_inputs_sliding    ['80%'] ,
                                                                        months_for_provided_inputs_sliding    ['20%']), axis=0)
        months_for_expected_outputs_sliding ['100%'] = np.concatenate( (months_for_expected_outputs_sliding   ['80%'] ,
                                                                        months_for_expected_outputs_sliding   ['20%']), axis=0)        
        #######################################################################
        self._assert_no_overlap_between_train_and_test(months_for_expected_outputs_tumbling, months_for_provided_inputs_tumbling, "tumbling")
        self._assert_no_overlap_between_train_and_test(months_for_expected_outputs_sliding , months_for_provided_inputs_sliding , "sliding")
        
        return (                  spei_dict         ,                months_dict           ,
                      spei_provided_inputs_sliding  , spei_expected_outputs_sliding        ,
                months_for_provided_inputs_sliding  , months_for_expected_outputs_sliding  ,
                      spei_provided_inputs_tumbling , spei_expected_outputs_tumbling       ,
                months_for_provided_inputs_tumbling , months_for_expected_outputs_tumbling )
    
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
    
    def _create_input_output_pairs(self, data_dict, configs_dict, data_label="data"):
        input_tumbling, output_tumbling = self._tumbling_window_maker(data_dict, configs_dict, data_label)
        input_sliding , output_sliding  = self._sliding_window_maker (data_dict, configs_dict, data_label)
        
        return input_tumbling, output_tumbling, input_sliding, output_sliding
    
    def _sliding_window_maker(self, data_dict, configs_dict, data_label="data"):
        
        sliding_window_len   = configs_dict['sliding_window_len'  ]
        sliding_window_step  = configs_dict['sliding_window_step' ]
        sliding_lookback_len = configs_dict['sliding_lookback_len']
        sliding_horizon_len  = configs_dict['sliding_horizon_len' ]
        
        input_sliding  = dict.fromkeys(Dataset.DATA_PORTION_TYPES)
        output_sliding = dict.fromkeys(Dataset.DATA_PORTION_TYPES)
        
        for data_portion_type in Dataset.DATA_PORTION_TYPES:
            # Data → sliding windows (with overlaps):
            windows_sliding = sliding_windower(x    = data_dict[data_portion_type],
                                       window_shape = sliding_window_len          )
            
            # reduces overlaps
            # Bad steps: 6, 3, 1.
            windows_sliding = windows_sliding[::sliding_window_step]
            
            input_sliding [data_portion_type] = windows_sliding[ : ,                      : sliding_lookback_len]
            output_sliding[data_portion_type] = windows_sliding[ : , -sliding_horizon_len :                     ]
            
            # +new dimension at the end of the array:
            input_sliding[data_portion_type] = input_sliding[data_portion_type][..., np.newaxis]
            
            # Debug:        
            # print("sliding_window_step =", sliding_window_step)
            # print("kept_window_indices =", list(range(0, min(len(windows_sliding), 5)*sliding_window_step, sliding_window_step)))
            
            # print(f'Sliding windows: {len(windows_sliding)}.')
            if data_label == "months":
                self._print_window_alignment_preview(
                    technique      = "sliding",
                    data_portion   = data_portion_type,
                    windows        = windows_sliding,
                    horizon_len    = sliding_horizon_len,
                    step           = sliding_window_step,
                    preview_count  = configs_dict.get("window_alignment_preview_count", 5),
                    enabled        = configs_dict.get("print_window_alignment_preview", False)
                )
            
        return input_sliding, output_sliding
    
    def _tumbling_window_maker(self, data_dict, configs_dict, data_label="data"):
        tumbling_window_len   = configs_dict['tumbling_window_len'  ]
        tumbling_window_step  = configs_dict['tumbling_window_step'  ]
        tumbling_lookback_len = configs_dict['tumbling_lookback_len']
        tumbling_horizon_len  = configs_dict['tumbling_horizon_len' ]
        
        input_tumbling  = dict.fromkeys(Dataset.DATA_PORTION_TYPES)
        output_tumbling = dict.fromkeys(Dataset.DATA_PORTION_TYPES)
        
        for data_portion_type in Dataset.DATA_PORTION_TYPES:
            # Data → sliding windows (with overlaps):
            windows_sliding = sliding_windower(x    = data_dict[data_portion_type],
                                       window_shape = tumbling_window_len         )
            
            # -overlaps by selecting only every 'tumbling_window_len'-th window:
            windows_tumbling = windows_sliding[::tumbling_window_step]
            
            input_tumbling [data_portion_type] = windows_tumbling[ : ,                       : tumbling_lookback_len]
            output_tumbling[data_portion_type] = windows_tumbling[ : , -tumbling_horizon_len :                      ]
            
            # +new dimension at the end of the array:
            input_tumbling[data_portion_type] = input_tumbling[data_portion_type][..., np.newaxis]
            
            # print(f'Tumbling windows: {len(windows_tumbling)}.')
            if data_label == "months":
                self._print_window_alignment_preview(
                    technique      = "tumbling",
                    data_portion   = data_portion_type,
                    windows        = windows_tumbling,
                    horizon_len    = tumbling_horizon_len,
                    step           = tumbling_window_step,
                    preview_count  = configs_dict.get("window_alignment_preview_count", 5),
                    enabled        = configs_dict.get("print_window_alignment_preview", False)
                )
            
        return input_tumbling, output_tumbling
    
    def _print_window_alignment_preview(self, technique, data_portion, windows, horizon_len, step, preview_count=5, enabled=False):
        if not enabled or len(windows) == 0:
            return
        
        print(f"[WINDOW-AUDIT] {self.city_name} | {technique} | {data_portion} | step={step}")
        limit = min(preview_count, len(windows))
        
        for idx in range(limit):
            window = windows[idx]
            if len(window) < horizon_len:
                print(f"  #{idx}: skipped (window_len={len(window)} < horizon_len={horizon_len})")
                continue
            window_start = window[0]
            window_end = window[-1]
            target_time = window[-horizon_len]
            print(f"  #{idx}: window_start={window_start}, window_end={window_end}, target_time={target_time}")
    
    def _validate_window_configs(self, configs_dict):
        window_pairs = (
            ("tumbling", configs_dict["tumbling_window_len"], configs_dict["tumbling_lookback_len"], configs_dict["tumbling_horizon_len"]),
            ("sliding" , configs_dict["sliding_window_len"] , configs_dict["sliding_lookback_len"] , configs_dict["sliding_horizon_len"] ),
        )
        
        for technique, window_len, lookback_len, horizon_len in window_pairs:
            if lookback_len + horizon_len != window_len:
                raise ValueError(
                    f"Invalid {technique} config: lookback ({lookback_len}) + horizon ({horizon_len}) "
                    f"must equal window_len ({window_len})."
                )
    
    def _assert_no_overlap_between_train_and_test(self, months_for_expected_outputs, months_for_provided_inputs, technique):
        if len(months_for_expected_outputs["80%"]) == 0 or len(months_for_provided_inputs["20%"]) == 0:
            return
        
        # expected outputs shape: (num_windows, horizon_len)
        last_train_window_targets = months_for_expected_outputs["80%"][-1]
        last_train_target_time = last_train_window_targets[-1]
        
        # provided inputs shape: (num_windows, lookback_len, 1)
        first_test_window_inputs = months_for_provided_inputs["20%"][0]
        if first_test_window_inputs.ndim < 2 or first_test_window_inputs.shape[0] == 0 or first_test_window_inputs.shape[1] == 0:
            raise ValueError(
                f"Invalid input window shape in {technique}: expected at least (1, 1), got {first_test_window_inputs.shape}."
            )
        first_test_input_time = first_test_window_inputs[0][0]
        
        if last_train_target_time >= first_test_input_time:
            raise ValueError(
                f"Potential train/test temporal overlap in {technique}: "
                f"last_train_target_time={last_train_target_time}, first_test_input_time={first_test_input_time}."
            )
