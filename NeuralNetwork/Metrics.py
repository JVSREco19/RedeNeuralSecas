import tensorflow as tf
import numpy      as np

def getError(actual, prediction):
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

def getTaylorMetrics(SPEI_dict, dataTrueValues_dict, predictValues_dict):    
    # Standard Deviation:
    predictions_std_dev       = {'Train': np.std(predictValues_dict['Train']),
                                 'Test' : np.std(predictValues_dict['Test' ])}
    
    combined_data             = np.concatenate([SPEI_dict['Train'], SPEI_dict['Test']])
    observed_std_dev          = np.std(combined_data)
    
    print(f"\t\t\tTRAIN: STD Dev {predictions_std_dev['Train']}")
    print(f"\t\t\tTEST : STD Dev {predictions_std_dev['Test' ]}")
    
    # Correlation Coefficient:
    correlation_coefficient  = {'Train': np.corrcoef(predictValues_dict['Train'], dataTrueValues_dict['Train'])[0, 1],
                                'Test' : np.corrcoef(predictValues_dict['Test' ], dataTrueValues_dict['Test' ])[0, 1]}
    
    print(f"\t\t\tTRAIN: correlation {correlation_coefficient['Train']}")
    print(f"\t\t\tTEST : correlation {correlation_coefficient['Test' ]}")
    
    return observed_std_dev, predictions_std_dev, correlation_coefficient