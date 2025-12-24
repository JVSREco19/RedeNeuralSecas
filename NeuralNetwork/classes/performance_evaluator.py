import tensorflow as tf
import numpy      as np
import pandas     as pd
from decimal import Decimal, InvalidOperation, ROUND_DOWN

class PerformanceEvaluator():
    
    # Constants for decimal comparison precision
    DECIMAL_PRECISION = 4  # Number of decimal digits to compare
    DECIMAL_QUANTIZE = Decimal('0.0001')  # Quantization for 4 decimal places
    DECIMAL_PADDING = '0000'  # Padding string for decimal digits
    
    def __init__(self):
        COLS_CENTRAL = {
            'Agrupamento'               : str  ,
            'Municipio Treinado'        : str  ,
            'Municipio Previsto'        : str  ,
            # 80% portion - Numpy
            'MAE 80% Numpy'             : float,
            'RMSE 80% Numpy'            : float,
            'MSE 80% Numpy'             : float,
            'R^2 80% Numpy'             : float,
            # 80% portion - Keras
            'MAE 80% Keras'             : float,
            'RMSE 80% Keras'            : float,
            'MSE 80% Keras'             : float,
            'R^2 80% Keras'             : float,
            # 80% portion - Comparisons
            'MAE 80% sign_equal'        : bool ,
            'MAE 80% integer_equal'     : bool ,
            'MAE 80% first4_equal'      : bool ,
            'RMSE 80% sign_equal'       : bool ,
            'RMSE 80% integer_equal'    : bool ,
            'RMSE 80% first4_equal'     : bool ,
            'MSE 80% sign_equal'        : bool ,
            'MSE 80% integer_equal'     : bool ,
            'MSE 80% first4_equal'      : bool ,
            'R^2 80% sign_equal'        : bool ,
            'R^2 80% integer_equal'     : bool ,
            'R^2 80% first4_equal'      : bool ,
            # 20% portion - Numpy
            'MAE 20% Numpy'             : float,
            'RMSE 20% Numpy'            : float,
            'MSE 20% Numpy'             : float,
            'R^2 20% Numpy'             : float,
            # 20% portion - Keras
            'MAE 20% Keras'             : float,
            'RMSE 20% Keras'            : float,
            'MSE 20% Keras'             : float,
            'R^2 20% Keras'             : float,
            # 20% portion - Comparisons
            'MAE 20% sign_equal'        : bool ,
            'MAE 20% integer_equal'     : bool ,
            'MAE 20% first4_equal'      : bool ,
            'RMSE 20% sign_equal'       : bool ,
            'RMSE 20% integer_equal'    : bool ,
            'RMSE 20% first4_equal'     : bool ,
            'MSE 20% sign_equal'        : bool ,
            'MSE 20% integer_equal'     : bool ,
            'MSE 20% first4_equal'      : bool ,
            'R^2 20% sign_equal'        : bool ,
            'R^2 20% integer_equal'     : bool ,
            'R^2 20% first4_equal'      : bool
        }
        
        COLS_BORDERING = {
            'Agrupamento'               : str  ,
            'Municipio Treinado'        : str  ,
            'Municipio Previsto'        : str  ,
            # 20% portion - Numpy
            'MAE 20% Numpy'             : float,
            'RMSE 20% Numpy'            : float,
            'MSE 20% Numpy'             : float,
            'R^2 20% Numpy'             : float,
            # 20% portion - Keras
            'MAE 20% Keras'             : float,
            'RMSE 20% Keras'            : float,
            'MSE 20% Keras'             : float,
            'R^2 20% Keras'             : float,
            # 20% portion - Comparisons
            'MAE 20% sign_equal'        : bool ,
            'MAE 20% integer_equal'     : bool ,
            'MAE 20% first4_equal'      : bool ,
            'RMSE 20% sign_equal'       : bool ,
            'RMSE 20% integer_equal'    : bool ,
            'RMSE 20% first4_equal'     : bool ,
            'MSE 20% sign_equal'        : bool ,
            'MSE 20% integer_equal'     : bool ,
            'MSE 20% first4_equal'      : bool ,
            'R^2 20% sign_equal'        : bool ,
            'R^2 20% integer_equal'     : bool ,
            'R^2 20% first4_equal'      : bool
        }
        
        self.metrics_central   = pd.DataFrame({col: pd.Series(dtype=typ) for col, typ in COLS_CENTRAL  .items()})
        self.metrics_bordering = pd.DataFrame({col: pd.Series(dtype=typ) for col, typ in COLS_BORDERING.items()})
    
    def _get_error_numpy(self, actual, prediction):
        """
        Deterministic NumPy-based metric computation.
        Computes MAE, MSE, RMSE and global (pooled) R^2 from the same masked residuals.
        Requires identical shapes, masks non-finite entries elementwise, flattens, computes metrics.
        Returns dict {'MAE','MSE','RMSE','R^2'}.
        """
        actual = np.asarray(actual)
        prediction = np.asarray(prediction)
        
        if actual.shape != prediction.shape:
            raise ValueError(f"Shape mismatch: actual {actual.shape} vs prediction {prediction.shape}")
        
        # Mask non-finite entries
        mask = np.isfinite(actual) & np.isfinite(prediction)
        actual_masked = actual[mask]
        prediction_masked = prediction[mask]
        
        # Flatten
        actual_flat = actual_masked.flatten()
        prediction_flat = prediction_masked.flatten()
        
        if len(actual_flat) == 0:
            return {'MAE': np.nan, 'MSE': np.nan, 'RMSE': np.nan, 'R^2': np.nan}
        
        # Compute residuals
        residuals = actual_flat - prediction_flat
        
        # MAE
        mae = np.mean(np.abs(residuals))
        
        # MSE
        mse = np.mean(residuals ** 2)
        
        # RMSE
        rmse = np.sqrt(mse)
        
        # R^2 (global/pooled)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((actual_flat - np.mean(actual_flat)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
        
        return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R^2': r2}
    
    def _get_error_keras(self, actual, prediction):
        """
        Keras-based metric computation that mirrors the NumPy behavior.
        Uses tf.keras.metrics and sample_weight to mask invalid elements.
        Returns dict {'MAE','MSE','RMSE','R^2'}.
        Exception-safe: returns NaNs on TF errors so pipeline continues.
        """
        try:
            actual = np.asarray(actual)
            prediction = np.asarray(prediction)
            
            # Store original shape
            original_shape = actual.shape
            
            # Flatten arrays to ensure consistent shapes
            actual_flat = actual.flatten()
            prediction_flat = prediction.flatten()
            
            # Create sample weights to mask non-finite values
            mask = np.isfinite(actual_flat) & np.isfinite(prediction_flat)
            sample_weight = mask.astype(np.float32)
            
            # If all weights are zero, return NaNs
            if np.sum(sample_weight) == 0:
                return {'MAE': np.nan, 'MSE': np.nan, 'RMSE': np.nan, 'R^2': np.nan}
            
            # Replace NaN/inf with zeros (they will be masked by sample_weight)
            actual_clean = np.where(mask, actual_flat, 0.0)
            prediction_clean = np.where(mask, prediction_flat, 0.0)
            
            # Reshape to 2D arrays with one feature dimension for Keras 3.x compatibility
            # Shape: (n_samples, 1)
            actual_clean = actual_clean.reshape(-1, 1)
            prediction_clean = prediction_clean.reshape(-1, 1)
            
            metrics = {
                'MAE'  : tf.keras.metrics.MeanAbsoluteError(),
                'MSE'  : tf.keras.metrics.MeanSquaredError(),
                'RMSE' : tf.keras.metrics.RootMeanSquaredError(),
                # Note: R2Score with variance_weighted_average may differ slightly from NumPy's 
                # global pooled R² computation, especially for multidimensional outputs.
                # This is expected and the comparison columns help identify differences.
                'R^2'  : tf.keras.metrics.R2Score(class_aggregation='variance_weighted_average')
            }
            
            metrics_values = {}
            
            for metric_name, metric_function in metrics.items():
                metric_function.update_state(actual_clean, prediction_clean, sample_weight=sample_weight)
                metrics_values[metric_name] = float(metric_function.result().numpy())
                metric_function.reset_state()
            
            return metrics_values
        
        except Exception as e:
            # Exception-safe: return NaNs on TF errors so pipeline continues
            return {'MAE': np.nan, 'MSE': np.nan, 'RMSE': np.nan, 'R^2': np.nan}
    
    def _to_decimal(self, value):
        """
        Convert a numeric value to Decimal, handling NaN/inf.
        Returns None for non-finite values.
        """
        try:
            if not np.isfinite(value):
                return None
            return Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError):
            return None
    
    def _extract_sign_integer_first4(self, value):
        """
        Extract sign ('+' or '-'), integer part (absolute), and first 4 decimal digits (truncated, zero-padded).
        Returns tuple (sign_str, int_part, first4_str) or (None, None, None) for non-finite values.
        """
        dec = self._to_decimal(value)
        if dec is None:
            return (None, None, None)
        
        # Sign
        sign_str = '+' if dec >= 0 else '-'
        
        # Work with absolute value
        dec_abs = abs(dec)
        
        # Integer part
        int_part = int(dec_abs)
        
        # Decimal part: get first 4 digits after decimal point
        # Truncate (round down) to precision defined by DECIMAL_QUANTIZE
        dec_truncated = dec_abs.quantize(self.DECIMAL_QUANTIZE, rounding=ROUND_DOWN)
        dec_frac = dec_truncated - int(dec_truncated)
        
        # Convert fractional part to string and extract digits
        frac_str = str(dec_frac)
        if '.' in frac_str:
            frac_digits = frac_str.split('.')[1]
        else:
            frac_digits = ''
        
        # Pad or truncate to exactly DECIMAL_PRECISION digits
        first4_str = (frac_digits + self.DECIMAL_PADDING)[:self.DECIMAL_PRECISION]
        
        return (sign_str, int_part, first4_str)
    
    def _compare_three_parts(self, numpy_val, keras_val):
        """
        Compare two values by three criteria:
        - sign ('+' or '-')
        - integer part (absolute integer part)
        - first four decimal digits after decimal point (truncated, zero-padded)
        Returns dict with keys: 'sign_equal', 'integer_equal', 'first4_equal' (bool or None if comparison not possible).
        """
        np_parts = self._extract_sign_integer_first4(numpy_val)
        keras_parts = self._extract_sign_integer_first4(keras_val)
        
        if np_parts[0] is None or keras_parts[0] is None:
            return {'sign_equal': None, 'integer_equal': None, 'first4_equal': None}
        
        sign_equal = (np_parts[0] == keras_parts[0])
        integer_equal = (np_parts[1] == keras_parts[1])
        first4_equal = (np_parts[2] == keras_parts[2])
        
        return {'sign_equal': sign_equal, 'integer_equal': integer_equal, 'first4_equal': first4_equal}
        
    def evaluate          (self       , is_model, spei_dict            ,
                           spei_expected_outputs, spei_predicted_values,
                           city_cluster_name    , city_for_training    , city_for_predicting):
        
        errors_dict = self._print_errors(spei_expected_outputs, spei_predicted_values         ,
                                         city_for_training   , city_for_predicting           , is_model)
        self.writeErrors(errors_dict      , spei_dict        , is_model, spei_expected_outputs, spei_predicted_values,
                         city_cluster_name, city_for_training, city_for_predicting)
        
        return self.metrics_central, self.metrics_bordering
    
    def getError(self, actual, prediction):
        """
        Compute metrics using both NumPy and Keras implementations.
        Returns nested dict {'numpy': {...}, 'keras': {...}}.
        """
        numpy_metrics = self._get_error_numpy(actual, prediction)
        keras_metrics = self._get_error_keras(actual, prediction)
        
        return {'numpy': numpy_metrics, 'keras': keras_metrics}

    def _print_errors(self, spei_expected_outputs, spei_predicted_values, city_for_training, city_for_predicting, is_model):
    
        # RMSE, MSE, MAE, R²:
        if is_model:
            errors_dict = {
                '80%' : self.getError(spei_expected_outputs['80%'], spei_predicted_values['80%']),
                '20%' : self.getError(spei_expected_outputs['20%'], spei_predicted_values['20%'])
                          }
            print(f'\t\t--------------Result for model {city_for_training} applied to its own data---------------')
            print(f"\t\t\tTRAIN ( 80%) NumPy: {errors_dict['80%']['numpy']}")
            print(f"\t\t\tTRAIN ( 80%) Keras: {errors_dict['80%']['keras']}")
            print(f"\t\t\tTEST  ( 20%) NumPy: {errors_dict['20%']['numpy']}")
            print(f"\t\t\tTEST  ( 20%) Keras: {errors_dict['20%']['keras']}")
        else:
            errors_dict = {
                '20%' : self.getError(spei_expected_outputs['20%' ], spei_predicted_values['20%' ])
                          }
            print(f'\t\t--------------Result for model {city_for_training} applied to {city_for_predicting} data---------------')
            print(f"\t\t\tTEST ( 20%) NumPy: {errors_dict['20%']['numpy']}")
            print(f"\t\t\tTEST ( 20%) Keras: {errors_dict['20%']['keras']}")

        return errors_dict

    def writeErrors(self, errors_dict   , spei_dict            , is_model,
                    spei_expected_outputs, spei_predicted_values,
                    city_cluster_name   , city_for_training    , city_for_predicting):
        
        # observed_std_dev, predictions_std_dev, correlation_coefficient = self.getTaylorMetrics(spei_dict, spei_expected_outputs, spei_predicted_values, is_model)
        
        if is_model:
            # Extract metrics for 80% portion
            mae_80_numpy = errors_dict['80%']['numpy']['MAE']
            mae_80_keras = errors_dict['80%']['keras']['MAE']
            rmse_80_numpy = errors_dict['80%']['numpy']['RMSE']
            rmse_80_keras = errors_dict['80%']['keras']['RMSE']
            mse_80_numpy = errors_dict['80%']['numpy']['MSE']
            mse_80_keras = errors_dict['80%']['keras']['MSE']
            r2_80_numpy = errors_dict['80%']['numpy']['R^2']
            r2_80_keras = errors_dict['80%']['keras']['R^2']
            
            # Compare 80% metrics
            mae_80_cmp = self._compare_three_parts(mae_80_numpy, mae_80_keras)
            rmse_80_cmp = self._compare_three_parts(rmse_80_numpy, rmse_80_keras)
            mse_80_cmp = self._compare_three_parts(mse_80_numpy, mse_80_keras)
            r2_80_cmp = self._compare_three_parts(r2_80_numpy, r2_80_keras)
            
            # Extract metrics for 20% portion
            mae_20_numpy = errors_dict['20%']['numpy']['MAE']
            mae_20_keras = errors_dict['20%']['keras']['MAE']
            rmse_20_numpy = errors_dict['20%']['numpy']['RMSE']
            rmse_20_keras = errors_dict['20%']['keras']['RMSE']
            mse_20_numpy = errors_dict['20%']['numpy']['MSE']
            mse_20_keras = errors_dict['20%']['keras']['MSE']
            r2_20_numpy = errors_dict['20%']['numpy']['R^2']
            r2_20_keras = errors_dict['20%']['keras']['R^2']
            
            # Compare 20% metrics
            mae_20_cmp = self._compare_three_parts(mae_20_numpy, mae_20_keras)
            rmse_20_cmp = self._compare_three_parts(rmse_20_numpy, rmse_20_keras)
            mse_20_cmp = self._compare_three_parts(mse_20_numpy, mse_20_keras)
            r2_20_cmp = self._compare_three_parts(r2_20_numpy, r2_20_keras)
            
            row = {
                'Agrupamento'             : city_cluster_name,
                'Municipio Treinado'      : city_for_training,
                'Municipio Previsto'      : city_for_predicting,
                # 80% portion - Numpy
                'MAE 80% Numpy'           : mae_80_numpy,
                'RMSE 80% Numpy'          : rmse_80_numpy,
                'MSE 80% Numpy'           : mse_80_numpy,
                'R^2 80% Numpy'           : r2_80_numpy,
                # 80% portion - Keras
                'MAE 80% Keras'           : mae_80_keras,
                'RMSE 80% Keras'          : rmse_80_keras,
                'MSE 80% Keras'           : mse_80_keras,
                'R^2 80% Keras'           : r2_80_keras,
                # 80% portion - Comparisons
                'MAE 80% sign_equal'      : mae_80_cmp['sign_equal'],
                'MAE 80% integer_equal'   : mae_80_cmp['integer_equal'],
                'MAE 80% first4_equal'    : mae_80_cmp['first4_equal'],
                'RMSE 80% sign_equal'     : rmse_80_cmp['sign_equal'],
                'RMSE 80% integer_equal'  : rmse_80_cmp['integer_equal'],
                'RMSE 80% first4_equal'   : rmse_80_cmp['first4_equal'],
                'MSE 80% sign_equal'      : mse_80_cmp['sign_equal'],
                'MSE 80% integer_equal'   : mse_80_cmp['integer_equal'],
                'MSE 80% first4_equal'    : mse_80_cmp['first4_equal'],
                'R^2 80% sign_equal'      : r2_80_cmp['sign_equal'],
                'R^2 80% integer_equal'   : r2_80_cmp['integer_equal'],
                'R^2 80% first4_equal'    : r2_80_cmp['first4_equal'],
                # 20% portion - Numpy
                'MAE 20% Numpy'           : mae_20_numpy,
                'RMSE 20% Numpy'          : rmse_20_numpy,
                'MSE 20% Numpy'           : mse_20_numpy,
                'R^2 20% Numpy'           : r2_20_numpy,
                # 20% portion - Keras
                'MAE 20% Keras'           : mae_20_keras,
                'RMSE 20% Keras'          : rmse_20_keras,
                'MSE 20% Keras'           : mse_20_keras,
                'R^2 20% Keras'           : r2_20_keras,
                # 20% portion - Comparisons
                'MAE 20% sign_equal'      : mae_20_cmp['sign_equal'],
                'MAE 20% integer_equal'   : mae_20_cmp['integer_equal'],
                'MAE 20% first4_equal'    : mae_20_cmp['first4_equal'],
                'RMSE 20% sign_equal'     : rmse_20_cmp['sign_equal'],
                'RMSE 20% integer_equal'  : rmse_20_cmp['integer_equal'],
                'RMSE 20% first4_equal'   : rmse_20_cmp['first4_equal'],
                'MSE 20% sign_equal'      : mse_20_cmp['sign_equal'],
                'MSE 20% integer_equal'   : mse_20_cmp['integer_equal'],
                'MSE 20% first4_equal'    : mse_20_cmp['first4_equal'],
                'R^2 20% sign_equal'      : r2_20_cmp['sign_equal'],
                'R^2 20% integer_equal'   : r2_20_cmp['integer_equal'],
                'R^2 20% first4_equal'    : r2_20_cmp['first4_equal']
            }
        else:
            # Extract metrics for 20% portion
            mae_20_numpy = errors_dict['20%']['numpy']['MAE']
            mae_20_keras = errors_dict['20%']['keras']['MAE']
            rmse_20_numpy = errors_dict['20%']['numpy']['RMSE']
            rmse_20_keras = errors_dict['20%']['keras']['RMSE']
            mse_20_numpy = errors_dict['20%']['numpy']['MSE']
            mse_20_keras = errors_dict['20%']['keras']['MSE']
            r2_20_numpy = errors_dict['20%']['numpy']['R^2']
            r2_20_keras = errors_dict['20%']['keras']['R^2']
            
            # Compare 20% metrics
            mae_20_cmp = self._compare_three_parts(mae_20_numpy, mae_20_keras)
            rmse_20_cmp = self._compare_three_parts(rmse_20_numpy, rmse_20_keras)
            mse_20_cmp = self._compare_three_parts(mse_20_numpy, mse_20_keras)
            r2_20_cmp = self._compare_three_parts(r2_20_numpy, r2_20_keras)
            
            row = {
                'Agrupamento'             : city_cluster_name,
                'Municipio Treinado'      : city_for_training,
                'Municipio Previsto'      : city_for_predicting,
                # 20% portion - Numpy
                'MAE 20% Numpy'           : mae_20_numpy,
                'RMSE 20% Numpy'          : rmse_20_numpy,
                'MSE 20% Numpy'           : mse_20_numpy,
                'R^2 20% Numpy'           : r2_20_numpy,
                # 20% portion - Keras
                'MAE 20% Keras'           : mae_20_keras,
                'RMSE 20% Keras'          : rmse_20_keras,
                'MSE 20% Keras'           : mse_20_keras,
                'R^2 20% Keras'           : r2_20_keras,
                # 20% portion - Comparisons
                'MAE 20% sign_equal'      : mae_20_cmp['sign_equal'],
                'MAE 20% integer_equal'   : mae_20_cmp['integer_equal'],
                'MAE 20% first4_equal'    : mae_20_cmp['first4_equal'],
                'RMSE 20% sign_equal'     : rmse_20_cmp['sign_equal'],
                'RMSE 20% integer_equal'  : rmse_20_cmp['integer_equal'],
                'RMSE 20% first4_equal'   : rmse_20_cmp['first4_equal'],
                'MSE 20% sign_equal'      : mse_20_cmp['sign_equal'],
                'MSE 20% integer_equal'   : mse_20_cmp['integer_equal'],
                'MSE 20% first4_equal'    : mse_20_cmp['first4_equal'],
                'R^2 20% sign_equal'      : r2_20_cmp['sign_equal'],
                'R^2 20% integer_equal'   : r2_20_cmp['integer_equal'],
                'R^2 20% first4_equal'    : r2_20_cmp['first4_equal']
            }
        
        if is_model:
            self.metrics_central.loc[len(self.metrics_central)] = row
        else:
            self.metrics_bordering.loc[len(self.metrics_bordering)] = row

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
