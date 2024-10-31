import tensorflow as tf

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
    
    return (metrics_values)