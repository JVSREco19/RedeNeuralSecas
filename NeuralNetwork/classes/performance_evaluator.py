import tensorflow as tf
import numpy      as np
import pandas     as pd

class PerformanceEvaluator():

    def evaluate          (self, has_trained   , spei_dict          ,
                           dataTrueValues_dict , predictValues_dict ,
                           city_cluster_name   , city_for_training  , city_for_predicting):
        
        errors_dict = self._print_errors(dataTrueValues_dict, predictValues_dict, city_for_training, city_for_predicting, has_trained)
        self.writeErrors(errors_dict, spei_dict, dataTrueValues_dict, predictValues_dict, city_cluster_name, city_for_training, city_for_predicting)
        
        return self.metrics_df
    
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

    def _print_errors(self, dataTrueValues_dict, predictValues_dict, city_for_training, city_for_predicting, has_trained):
    
        match has_trained:
            case False:
                print(f'\t\t--------------Result for {city_for_training} (training)---------------')
            case True :
                print(f'\t\t--------------Result for {city_for_training} applied to {city_for_predicting}---------------')
            case _    :
                print('Error in method _print_errors of class PerformanceEvaluator: the has_trained state cannot be recognized.')
                return False
    
        # RMSE, MSE, MAE, R²:
        errors_dict = {
            'Train': self.getError(dataTrueValues_dict['Train'], predictValues_dict['Train']),
            'Test' : self.getError(dataTrueValues_dict['Test' ], predictValues_dict['Test' ])
                      }
    
        print(f"\t\t\tTRAIN: {errors_dict['Train']}")
        print(f"\t\t\tTEST : {errors_dict['Test'] }")
        
        return errors_dict

    def writeErrors(self, errors_dict  , spei_dict          ,
                    dataTrueValues_dict, predictValues_dict ,
                    city_cluster_name  , city_for_training  , city_for_predicting):
        observed_std_dev, predictions_std_dev, correlation_coefficient = self.getTaylorMetrics(spei_dict, dataTrueValues_dict, predictValues_dict)

        self.metrics_df = pd.DataFrame(columns=['Agrupamento', 'Municipio Treinado', 'Municipio Previsto', 'MAE Treinamento', 'MAE Validação', 'RMSE Treinamento', 'RMSE Validação', 'MSE Treinamento', 'MSE Validação', 'R^2 Treinamento', 'R^2 Validação', 'Desvio Padrão Obs.', 'Desvio Padrão Pred. Treinamento', 'Desvio Padrão Pred. Validação', 'Coef. de Correlação Treinamento', 'Coef. de Correlação Validação'])
        
        row = {
            'Agrupamento'                    : city_cluster_name                        ,
            'Municipio Treinado'             : city_for_training                        ,
            'Municipio Previsto'             : city_for_predicting                      ,
            'MAE Treinamento'                : errors_dict            ['Train']['MAE' ] ,
            'MAE Validação'                  : errors_dict            ['Test' ]['MAE' ] ,
            'RMSE Treinamento'               : errors_dict            ['Train']['RMSE'] ,
            'RMSE Validação'                 : errors_dict            ['Test' ]['RMSE'] ,
            'MSE Treinamento'                : errors_dict            ['Train']['MSE' ] ,
            'MSE Validação'                  : errors_dict            ['Test' ]['MSE' ] ,
            'R^2 Treinamento'                : errors_dict            ['Train']['R^2' ] ,
            'R^2 Validação'                  : errors_dict            ['Test' ]['R^2' ] ,
            'Desvio Padrão Obs.'             : observed_std_dev                         ,
            'Desvio Padrão Pred. Treinamento': predictions_std_dev    ['Train']         ,
            'Desvio Padrão Pred. Validação'  : predictions_std_dev    ['Test' ]         ,
            'Coef. de Correlação Treinamento': correlation_coefficient['Train']         ,
            'Coef. de Correlação Validação'  : correlation_coefficient['Test' ]
        }

        self.metrics_df = pd.concat([self.metrics_df, pd.DataFrame([row])], ignore_index=True)

    def getTaylorMetrics(self, spei_dict, dataTrueValues_dict, predictValues_dict):    
     # Standard Deviation:
     predictions_std_dev       = {'Train': np.std(predictValues_dict['Train']),
                                  'Test' : np.std(predictValues_dict['Test' ])}
     
     combined_data             = np.concatenate([spei_dict['Train'], spei_dict['Test']])
     observed_std_dev          = np.std(combined_data)
     
     print(f"\t\t\tTRAIN: STD Dev {predictions_std_dev['Train']}")
     print(f"\t\t\tTEST : STD Dev {predictions_std_dev['Test' ]}")
     
     # Correlation Coefficient:
     correlation_coefficient  = {'Train': np.corrcoef(predictValues_dict['Train'], dataTrueValues_dict['Train'])[0, 1],
                                 'Test' : np.corrcoef(predictValues_dict['Test' ], dataTrueValues_dict['Test' ])[0, 1]}
     
     print(f"\t\t\tTRAIN: correlation {correlation_coefficient['Train']}")
     print(f"\t\t\tTEST : correlation {correlation_coefficient['Test' ]}")
     
     return observed_std_dev, predictions_std_dev, correlation_coefficient