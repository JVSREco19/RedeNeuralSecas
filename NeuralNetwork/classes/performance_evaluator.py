import tensorflow as tf
import numpy      as np
import pandas     as pd

class PerformanceEvaluator():
    
    def __init__(self):
        COLS_CENTRAL = {
            'Agrupamento'               : str  ,
            'Municipio Treinado'        : str  ,
            'Municipio Previsto'        : str  ,
            'MAE 80%'                   : float,
            'MAE 20%'                   : float,
            'RMSE 80%'                  : float,
            'RMSE 20%'                  : float,
            'MSE 80%'                   : float,
            'MSE 20%'                   : float,
            'R^2 80%'                   : float,
            'R^2 20%'                   : float
            # 'Desvio Padrão Obs.'      : float,
            # 'Desvio Padrão Pred. 80%' : float,
            # 'Desvio Padrão Pred. 20%' : float,
            # 'Coef. de Correlação 80%' : float,
            # 'Coef. de Correlação 20%' : float
        }
        
        COLS_BORDERING = {
            'Agrupamento'               : str  ,
            'Municipio Treinado'        : str  ,
            'Municipio Previsto'        : str  ,
            'MAE 100%'                  : float,
            'MAE 20%'                   : float,
            'RMSE 100%'                 : float,
            'RMSE 20%'                  : float,
            'MSE 100%'                  : float,
            'MSE 20%'                   : float,
            'R^2 100%'                  : float,
            'R^2 20%'                   : float
            # 'Desvio Padrão Obs.'      : float,
            # 'Desvio Padrão Pred. 100%': float,
            # 'Desvio Padrão Pred. 20%' : float,
            # 'Coef. de Correlação 100%': float,
            # 'Coef. de Correlação 20%' : float
        }
        
        self.metrics_central   = pd.DataFrame({col: pd.Series(dtype=typ) for col, typ in COLS_CENTRAL  .items()})
        self.metrics_bordering = pd.DataFrame({col: pd.Series(dtype=typ) for col, typ in COLS_BORDERING.items()})
        
    def evaluate          (self       , is_model, spei_dict            ,
                           spei_expected_outputs, spei_predicted_values,
                           city_cluster_name    , city_for_training    , city_for_predicting):
        
        errors_dict = self._print_errors(spei_expected_outputs, spei_predicted_values         ,
                                         city_for_training   , city_for_predicting           , is_model)
        self.writeErrors(errors_dict      , spei_dict        , is_model, spei_expected_outputs, spei_predicted_values,
                         city_cluster_name, city_for_training, city_for_predicting)
        
        return self.metrics_central, self.metrics_bordering
    
    def getError(self, actual, prediction):
        metrics = {
            'RMSE' : tf.keras.metrics.RootMeanSquaredError(),
            'MSE'  : tf.keras.metrics.MeanSquaredError    (),
            'MAE'  : tf.keras.metrics.MeanAbsoluteError   (),
            'R^2'  : tf.keras.metrics.R2Score             (class_aggregation='variance_weighted_average')
        }
    
        metrics_values = dict.fromkeys(metrics.keys())
        
        for metric_name, metric_function in metrics.items():
            metric_function.update_state(actual, prediction)
            metrics_values[metric_name] = metric_function.result().numpy()
        
        return metrics_values

    def _print_errors(self, spei_expected_outputs, spei_predicted_values, city_for_training, city_for_predicting, is_model):
    
        # RMSE, MSE, MAE, R²:
        if is_model:
            errors_dict = {
                '80%' : self.getError(spei_expected_outputs['80%'], spei_predicted_values['80%']),
                '20%' : self.getError(spei_expected_outputs['20%'], spei_predicted_values['20%'])
                          }
            print(f'\t\t--------------Result for model {city_for_training} applied to its own data---------------')
            print(f"\t\t\tTRAIN ( 80%): {errors_dict['80%' ]}")
            print(f"\t\t\tTEST  ( 20%): {errors_dict['20%' ] }")
        else:
            errors_dict = {
                '100%': self.getError(spei_expected_outputs['100%'], spei_predicted_values['100%']),
                '20%' : self.getError(spei_expected_outputs['20%' ], spei_predicted_values['20%' ])
                          }
            print(f'\t\t--------------Result for model {city_for_training} applied to {city_for_predicting} data---------------')
            print(f"\t\t\tTEST (100%): {errors_dict['20%' ] }")
            print(f"\t\t\tTEST ( 20%): {errors_dict['100%'] }")

        return errors_dict

    def writeErrors(self, errors_dict   , spei_dict            , is_model,
                    spei_expected_outputs, spei_predicted_values,
                    city_cluster_name   , city_for_training    , city_for_predicting):
        
        # observed_std_dev, predictions_std_dev, correlation_coefficient = self.getTaylorMetrics(spei_dict, spei_expected_outputs, spei_predicted_values, is_model)
        
        if is_model:
            row = {
                'Agrupamento'             : city_cluster_name                       ,
                'Municipio Treinado'      : city_for_training                       ,
                'Municipio Previsto'      : city_for_predicting                     ,
                'MAE 80%'                 : errors_dict             [ '80%']['MAE' ] ,
                'MAE 20%'                 : errors_dict             [ '20%']['MAE' ] ,
                'RMSE 80%'                : errors_dict             [ '80%']['RMSE'] ,
                'RMSE 20%'                : errors_dict             [ '20%']['RMSE'] ,
                'MSE 80%'                 : errors_dict             [ '80%']['MSE' ] ,
                'MSE 20%'                 : errors_dict             [ '20%']['MSE' ] ,
                'R^2 80%'                 : errors_dict             [ '80%']['R^2' ] ,
                'R^2 20%'                 : errors_dict             [ '20%']['R^2' ]
                # 'Desvio Padrão Obs.'      : observed_std_dev                        ,
                # 'Desvio Padrão Pred. 80%' : predictions_std_dev     ['80%']         ,
                # 'Desvio Padrão Pred. 20%' : predictions_std_dev     ['20%']         ,
                # 'Coef. de Correlação 80%' : correlation_coefficient ['80%']         ,
                # 'Coef. de Correlação 20%' : correlation_coefficient ['20%']
            }
        else:
            row = {
                'Agrupamento'             : city_cluster_name                       ,
                'Municipio Treinado'      : city_for_training                       ,
                'Municipio Previsto'      : city_for_predicting                     ,
                'MAE 100%'                : errors_dict             ['100%']['MAE' ] ,
                'MAE 20%'                 : errors_dict             [ '20%']['MAE' ] ,
                'RMSE 100%'               : errors_dict             ['100%']['RMSE'] ,
                'RMSE 20%'                : errors_dict             [ '20%']['RMSE'] ,
                'MSE 100%'                : errors_dict             ['100%']['MSE' ] ,
                'MSE 20%'                 : errors_dict             [ '20%']['MSE' ] ,
                'R^2 100%'                : errors_dict             ['100%']['R^2' ] ,
                'R^2 20%'                 : errors_dict             [ '20%']['R^2' ]
                # 'Desvio Padrão Obs.'      : observed_std_dev                        ,
                # 'Desvio Padrão Pred. 80%' : predictions_std_dev     ['80%']         ,
                # 'Desvio Padrão Pred. 20%' : predictions_std_dev     ['20%']         ,
                # 'Coef. de Correlação 80%' : correlation_coefficient ['80%']         ,
                # 'Coef. de Correlação 20%' : correlation_coefficient ['20%']
            }
        
        if is_model:
            df_row = pd.DataFrame([row]).astype(self.metrics_central.dtypes.to_dict())
            self.metrics_central   = pd.concat([self.metrics_central  , df_row], ignore_index=True)
        else:
            df_row = pd.DataFrame([row]).astype(self.metrics_bordering.dtypes.to_dict())
            self.metrics_bordering = pd.concat([self.metrics_bordering, df_row], ignore_index=True)

    # def getTaylorMetrics(self, spei_dict, spei_expected_outputs, spei_predicted_values, is_model):    
    #  # Standard Deviation:
    #  if is_model:
    #      predictions_std_dev       = {'80%' : np.std(spei_predicted_values['80%']),
    #                                   '20%' : np.std(spei_predicted_values['20%'])}
     
    #      combined_data             = np.concatenate([spei_dict['80%'], spei_dict['20%']])
    #      observed_std_dev          = np.std(combined_data)
     
    #      print(f"\t\t\tTRAIN (80%): STD Dev {predictions_std_dev['80%']}")
    #      print(f"\t\t\tTEST  (20%): STD Dev {predictions_std_dev['20%']}")
     
    #      # Correlation Coefficient:
    #      correlation_coefficient  = {'80%' : np.corrcoef(spei_predicted_values['80%'], spei_expected_outputs['80%'])[0, 1],
    #                                  '20%' : np.corrcoef(spei_predicted_values['20%'], spei_expected_outputs['20%'])[0, 1]}
     
    #      print(f"\t\t\tTRAIN (80%): correlation {correlation_coefficient['80%']}")
    #      print(f"\t\t\tTEST  (20%): correlation {correlation_coefficient['20%']}")

    #      return observed_std_dev, predictions_std_dev, correlation_coefficient
    #  else:
    #      return None, None, None # Quick fix, not intended to be like that when using TayilorMetrics!
