import tensorflow as tf
import json
from .performance_evaluator import PerformanceEvaluator

class NeuralNetwork:

    DATA_TYPES_LIST = ['80%', '20%']

    def __init__(self, file_name, dataset, plotter):
        self.dataset        = dataset
        self.plotter        = plotter
        self.evaluator      = PerformanceEvaluator()
        
        self.configs_dict   = self._set_configs(file_name)
        self.model_sliding  = self._create_ml_model((self.configs_dict['sliding_lookback_len' ], 1))
        self.model_tumbling = self._create_ml_model((self.configs_dict['tumbling_lookback_len'], 1))
        self.has_trained    = False
        
        # print('Input shape:', self.model.input_shape)
        # print(self.model.summary())
    
    def _set_configs(self, file_name):
        with open(file_name) as file:
            configs_dict = json.load(file)
        
        configs_dict.update(
            {'activation'  : ['relu', 'sigmoid'],
             'loss'        : 'mse',
             'metrics'     : ['mae',
                             tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                             'mse',
                             tf.keras.metrics.R2Score(name="r2")],
             'optimizer'   : 'adam'
            }
       )
        
        return configs_dict        

    def _create_ml_model(self, input_shape):
        # print(f'Started: creation of ML model {self.dataset.city_name}')
        model = tf.keras.Sequential()
        model.add(tf.keras.Input       (    shape=input_shape                             )      )
        model.add(tf.keras.layers.LSTM (          self.configs_dict['hidden_units']       ,
                                       activation=self.configs_dict['activation'  ][0])   )
        for _ in range(3):
            model.add(tf.keras.layers.Dense(units=self.configs_dict['dense_units' ]       ,
                                       activation=self.configs_dict['activation'][1])     )
        model.compile(loss=self.configs_dict['loss'], metrics=self.configs_dict['metrics'],
                      optimizer=self.configs_dict['optimizer']                            )
        # print(f'Ended: creation of ML model {self.dataset.city_name}')
        
        return model
    
    def _train_ml_models(self, spei_provided_inputs_sliding   ,
                               spei_expected_outputs_sliding  ,
                               spei_provided_inputs_tumbling  ,
                               spei_expected_outputs_tumbling):
      
        print(f'\nStarted: training of ML model {self.dataset.city_name}, sliding windows (may take a BIG while)')
        history_sliding = self.model_sliding.fit(
            spei_provided_inputs_sliding  ['80%'],
            spei_expected_outputs_sliding ['80%'],
            epochs=self.configs_dict['numberOfEpochs'], batch_size=256, verbose=0)
        self.has_trained = True
        print(f'Ended  : training of ML model {self.dataset.city_name}, sliding windows' )
        
        print(f'\nStarted: training of ML model {self.dataset.city_name}, tumbling windows (may take a while)')
        history_tumbling = self.model_tumbling.fit(
            spei_provided_inputs_tumbling  ['80%'],
            spei_expected_outputs_tumbling ['80%'],
            epochs=self.configs_dict['numberOfEpochs'], batch_size=1, verbose=0)
        self.has_trained = True
        print(f'Ended  : training of ML model {self.dataset.city_name}, tumbling windows')
        
        return history_sliding, history_tumbling #history_tumbling, history_tumbling
    
    def use_neural_network(self, dataset=None, plotter=None):
        if plotter == None: plotter = self.plotter
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
            history_sliding, history_tumbling = self._train_ml_models(spei_provided_inputs_sliding  ,
                                                                      spei_expected_outputs_sliding ,
                                                                      spei_provided_inputs_tumbling ,
                                                                      spei_expected_outputs_tumbling)
            
            plotter.drawModelLineGraph           (history_sliding , self.dataset.city_cluster_name, self.dataset.city_name)
            plotter.drawModelLineGraph           (history_tumbling, self.dataset.city_cluster_name, self.dataset.city_name)
            
        print(f'Started: applying ML model {self.dataset.city_name} to city {dataset.city_name}')
        
        if is_model:
            print(f'Is model? {is_model}.')
            
            print('STARTED making predictions for Sliding Windows')
            spei_predicted_values_sliding = {
                '80%' : self.model_sliding.predict(spei_provided_inputs_sliding  ['80%'], verbose = 0),
                '20%' : self.model_sliding.predict(spei_provided_inputs_sliding  ['20%'], verbose = 0)
                                    }
            print('ENDED making predictions for Sliding Windows')
            
            print('STARTED making predictions for Tumbling Windows')
            spei_predicted_values_tumbling = {
                '80%' : self.model_tumbling.predict(spei_provided_inputs_tumbling['80%'], verbose = 0),
                '20%' : self.model_tumbling.predict(spei_provided_inputs_tumbling['20%'], verbose = 0)
                                    }
            print('ENDED making predictions for Tumbling Windows')
            
        else:
            print(f'Is model? {is_model}.')
            
            print('STARTED making predictions for Sliding Windows')
            spei_predicted_values_sliding = {
                '20%' : self.model_sliding.predict(spei_provided_inputs_sliding  ['20%'], verbose = 0)
                                    }
            print('ENDED making predictions for Sliding Windows')
            
            print('STARTED making predictions for Tumbling Windows')
            spei_predicted_values_tumbling = {
                '20%' : self.model_tumbling.predict(spei_provided_inputs_tumbling['20%'], verbose = 0)
                                    }
            print('ENDED making predictions for Tumbling Windows')
       
        # BUG: metrics_central_sliding and metrics_central_tumbling are being written with the same mixed content.
        metrics_central_sliding, metrics_bordering_sliding = self.evaluator.evaluate('Sliding', is_model, spei_dict,
            spei_expected_outputs_sliding , spei_predicted_values_sliding               ,
            self.dataset.city_cluster_name, self.dataset.city_name , dataset.city_name  )
        
        # BUG: metrics_central_sliding and metrics_central_tumbling are being written with the same mixed content.
        metrics_central_tumbling, metrics_bordering_tumbling = self.evaluator.evaluate('Tumbling', is_model, spei_dict,
            spei_expected_outputs_tumbling, spei_predicted_values_tumbling              ,
            self.dataset.city_cluster_name, self.dataset.city_name , dataset.city_name  )
        
        
        plotter.plotDatasetPlots   (dataset, spei_dict['20%']      , split_position   ,
            self.dataset.city_cluster_name , self.dataset.city_name, dataset.city_name)
        
        self.plotter.plotModelPlots(dataset, spei_dict, is_model                       ,
            spei_expected_outputs_sliding            , spei_predicted_values_sliding   ,
            months_for_expected_outputs_sliding      , self.has_trained                ,
            history_sliding if not self.has_trained else None                          ,
            metrics_central_sliding if is_model     else metrics_bordering_sliding     ,
            self.dataset.city_cluster_name, self.dataset.city_name  , dataset.city_name, 'Sliding Windows')
        
        self.plotter.plotModelPlots(dataset, spei_dict, is_model                       ,
            spei_expected_outputs_tumbling            , spei_predicted_values_tumbling ,
            months_for_expected_outputs_tumbling      , self.has_trained               ,
            history_tumbling if not self.has_trained else None                         ,
            metrics_central_tumbling if is_model     else metrics_bordering_tumbling   ,
            self.dataset.city_cluster_name, self.dataset.city_name  , dataset.city_name, 'Tumbling Windows')
        
        print(f'Ended  : applying ML model {self.dataset.city_name} to city {dataset.city_name}')
        
        return metrics_central_sliding, metrics_bordering_sliding, metrics_central_tumbling, metrics_bordering_tumbling