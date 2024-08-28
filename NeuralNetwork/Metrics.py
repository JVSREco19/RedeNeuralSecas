import tensorflow as tf

def r_square(y_true, y_pred):
  from sklearn.metrics import r2_score
  r2 = r2_score(y_true, y_pred, multioutput='variance_weighted')
  return r2

def getError(actual, prediction):
  rmse = tf.keras.metrics.RootMeanSquaredError()
  mse = tf.keras.metrics.MeanSquaredError()
  mae = tf.keras.metrics.MeanAbsoluteError()

  rmse.update_state(actual, prediction)
  mse.update_state(actual, prediction)
  mae.update_state(actual, prediction)

  err1 = mae.result().numpy()
  err2 = rmse.result().numpy()
  err3 = mse.result().numpy()
  err4 = r_square(actual, prediction)

  return ({'MAE':err1, "RMSE" : err2, 'MSE': err3, 'R^2': err4 })