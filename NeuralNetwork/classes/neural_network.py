import tensorflow as tf
import json
from .performance_evaluator import PerformanceEvaluator

class NeuralNetwork:

    DATA_TYPES_LIST = ['80%', '20%']

    def __init__(self, file_name, dataset, plotter):
        self.dataset        = dataset
        # self.plotter        = plotter
        self.evaluator      = PerformanceEvaluator()
        
        self.configs_dict   = self._set_configs(file_name)
        
        self.model_tumbling = self._create_ml_model('tumbling')
        self.model_sliding  = self._create_ml_model('sliding' )
        
        self.has_trained    = False
        
        # print(f"input_shape_sliding : {self.configs_dict['input_shape_sliding' ]};")
        # print(f"input_shape_tumbling: {self.configs_dict['input_shape_tumbling']}.")
        
        # print('Input shape:', self.model.input_shape)
        # print(self.model.summary())
    
    def _set_configs(self, file_name):
        with open(file_name) as file:
            configs_dict = json.load(file)
        
        configs_dict.update(
            {'input_shape_sliding' : (configs_dict['sliding_lookback_len' ], 1),
             'input_shape_tumbling': (configs_dict['tumbling_lookback_len'], 1),
             'activation'  : ['relu', 'sigmoid'],
             'loss'        : 'mse',
             'metrics'     : ['mae',
                             tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                             'mse',
                             tf.keras.metrics.R2Score(name="r2")],
             'optimizer'   : 'adam'
            }
       )
        
        return configs_dict        

    def _create_ml_model(self, technique):
        # print(f'Started: creation of ML model {self.dataset.city_name}')
        model = tf.keras.Sequential()       
        model.add(tf.keras.Input           (    shape  = self.configs_dict[f'input_shape_{technique}']) )
        model.add(tf.keras.layers.LSTM     (             self.configs_dict[f'{technique}_hidden_units']              ,
                                            activation = self.configs_dict[ 'activation'             ][0])          )

        for _ in range(self.configs_dict[f'{technique}_num_dense_layers']):
            model.add(tf.keras.layers.Dense(     units = self.configs_dict[f'{technique}_dense_units'],
                                            activation = self.configs_dict["activation"              ][1]))
        
        model.add    (tf.keras.layers.Dense(     units = self.configs_dict[f"{technique}_horizon_len"],
                                            activation = "linear"))
            
        model.compile(loss      = self.configs_dict['loss'     ],
                      metrics   = self.configs_dict['metrics'  ],
                      optimizer = self.configs_dict['optimizer'])
        # print(f'Ended: creation of ML model {self.dataset.city_name}')
        
        return model
    
    def _train_ml_models(self, spei_provided_inputs_sliding   ,
                               spei_expected_outputs_sliding  ,
                               spei_provided_inputs_tumbling  ,
                               spei_expected_outputs_tumbling):
        
        print(f'\nStarted: training of ML model {self.dataset.city_name}, tumbling windows (may take a while)')
        # print(spei_provided_inputs_tumbling  ['80%'].shape)
        # print(spei_expected_outputs_tumbling ['80%'].shape)
        
        history_tumbling = self.model_tumbling.fit(
            spei_provided_inputs_tumbling  ['80%'],
            spei_expected_outputs_tumbling ['80%'],
            epochs=self.configs_dict['numberOfEpochs'], batch_size= 1, verbose=0)
        self.has_trained = True
        print(f'Ended  : training of ML model {self.dataset.city_name}, tumbling windows')
        
        print(f'\nStarted: training of ML model {self.dataset.city_name}, sliding windows (may take a BIG while)')
        # print(spei_provided_inputs_sliding  ['80%'].shape)
        # print(spei_expected_outputs_sliding ['80%'].shape)
        
        history_sliding = self.model_sliding.fit(
            spei_provided_inputs_sliding  ['80%'],
            spei_expected_outputs_sliding ['80%'],
            epochs=self.configs_dict['numberOfEpochs'], batch_size= 8, verbose=0)
        self.has_trained = True
        print(f'Ended  : training of ML model {self.dataset.city_name}, sliding windows' )
        
        return history_tumbling, history_sliding
    
    def use_neural_network(self, dataset=None, plotter=None):
        # if plotter == None: plotter = self.plotter
        if dataset == None:
              dataset  = self.dataset
              is_model = True
        else: is_model = False
        
        # For bordering cities, use the training dataset's normalization parameters
        if is_model:
            (spei_dict                           ,                months_dict           ,
             spei_provided_inputs_sliding        , spei_expected_outputs_sliding        ,
             months_for_provided_inputs_sliding  , months_for_expected_outputs_sliding  ,
             spei_provided_inputs_tumbling       , spei_expected_outputs_tumbling       ,
             months_for_provided_inputs_tumbling , months_for_expected_outputs_tumbling ) = dataset.format_data_for_model(self.configs_dict)
            
        else:
            (spei_dict                           ,                months_dict           ,
             spei_provided_inputs_sliding        , spei_expected_outputs_sliding        ,
             months_for_provided_inputs_sliding  , months_for_expected_outputs_sliding  ,
             spei_provided_inputs_tumbling       , spei_expected_outputs_tumbling       ,
             months_for_provided_inputs_tumbling , months_for_expected_outputs_tumbling ) = dataset.format_data_for_model(
                 self.configs_dict, self.dataset.spei_min, self.dataset.spei_max)
       
        split_position = len(spei_dict['80%'])
        
        if not self.has_trained:
            # flags has_trained as True:
            history_tumbling, history_sliding  = self._train_ml_models(spei_provided_inputs_sliding  ,
                                                                      spei_expected_outputs_sliding ,
                                                                      spei_provided_inputs_tumbling ,
                                                                      spei_expected_outputs_tumbling)
            # plotter.drawModelLineGraph           (history, self.dataset.city_cluster_name, self.dataset.city_name)
            
        print(f'Started: applying ML model {self.dataset.city_name} to city {dataset.city_name}')
        
        if is_model:
            print(f'Is model? {is_model}.')
            
            print('STARTED making predictions for Tumbling Windows')
            spei_predicted_values_tumbling = {
                '80%' : self.model_tumbling.predict(spei_provided_inputs_tumbling['80%'], verbose = 0),
                '20%' : self.model_tumbling.predict(spei_provided_inputs_tumbling['20%'], verbose = 0)
                                    }
            print('ENDED making predictions for Tumbling Windows')
            
            print('STARTED making predictions for Sliding Windows')
            spei_predicted_values_sliding = {
                '80%' : self.model_sliding.predict(spei_provided_inputs_sliding  ['80%'], verbose = 0),
                '20%' : self.model_sliding.predict(spei_provided_inputs_sliding  ['20%'], verbose = 0)
                                    }
            print('ENDED making predictions for Sliding Windows')
            
        else:
            print(f'Is model? {is_model}.')
            
            print('STARTED making predictions for Tumbling Windows')
            spei_predicted_values_tumbling = {
                '20%' : self.model_tumbling.predict(spei_provided_inputs_tumbling['20%'], verbose = 0)
                                    }
            print('ENDED making predictions for Tumbling Windows')

            print('STARTED making predictions for Sliding Windows')
            spei_predicted_values_sliding = {
                '20%' : self.model_sliding.predict(spei_provided_inputs_sliding  ['20%'], verbose = 0)
                                    }
            print('ENDED making predictions for Sliding Windows')
                     
        metrics_central_tumbling, metrics_bordering_tumbling = self.evaluator.evaluate('tumbling',
            is_model                      , spei_dict                                            ,
            spei_expected_outputs_tumbling, spei_predicted_values_tumbling                       ,
            self.dataset.city_cluster_name, self.dataset.city_name , dataset.city_name           )
        
        metrics_central_sliding, metrics_bordering_sliding = self.evaluator.evaluate('sliding',
            is_model                      , spei_dict                                         ,
            spei_expected_outputs_sliding , spei_predicted_values_sliding                     ,
            self.dataset.city_cluster_name, self.dataset.city_name , dataset.city_name        )
        
        # plotter.plotDatasetPlots   (dataset, spei_dict['20%']      , split_position   ,
        #     self.dataset.city_cluster_name , self.dataset.city_name, dataset.city_name)
        
        # self.plotter.plotModelPlots(dataset, spei_dict, is_model             ,
        #     spei_expected_outputs_tumbling            , spei_predicted_values,
        #     months_for_expected_outputs_tumbling      , self.has_trained     ,
        #     history if not self.has_trained else None               ,
        #     metrics_central if is_model     else metrics_bordering  ,
        #     self.dataset.city_cluster_name, self.dataset.city_name  , dataset.city_name)
        
        print(f'Ended  : applying ML model {self.dataset.city_name} to city {dataset.city_name}')
        
        return (metrics_central_tumbling, metrics_bordering_tumbling,
                metrics_central_sliding , metrics_bordering_sliding )