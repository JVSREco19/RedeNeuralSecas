import os
import shutil

import tensorflow as tf
import pandas     as pd

from NeuralNetwork.classes       import NeuralNetwork  , Plotter
from NeuralNetworkDriver.classes import InputDataLoader

INPUT_DIR_ADDR    = './Data/'
OUTPUT_DIR_ADDR   = './Output/'
NEURAL_NET_CONFIG = './NeuralNetwork/config.json'

def make_output_dirs(rootdir, clusters):
    # Clears ouput directory before starting:
    if os.path.isdir (rootdir):
        shutil.rmtree(rootdir)
    
    for cluster_name, cities_dict in clusters.items():
        for city in cities_dict.keys():
            os.makedirs(f'{rootdir}/cluster {cluster_name}/model {cluster_name}/city {city}')

def instantiate_ml_models_for_central_cities():
    neural_network_models = dict.fromkeys(clusters)
    
    for central_city in neural_network_models.keys():
        DATASET = clusters[central_city][central_city]
        neural_network_models[central_city] = NeuralNetwork(NEURAL_NET_CONFIG, DATASET, THE_PLOTTER)
        print(f'\tCreated ML model {central_city}')
        
        tf.keras.backend.clear_session()
        
    return neural_network_models

def train_ml_models_for_central_cities():
    metrics_df_central_cities = None
    
    for neural_network_model_name, neural_network_model in neural_network_models.items():
        metrics_df_current_central_city, _ = neural_network_model.use_neural_network()

        if metrics_df_central_cities is None or metrics_df_central_cities.empty:
            metrics_df_central_cities = metrics_df_current_central_city
        else:
            metrics_df_central_cities = pd.concat([metrics_df_central_cities, metrics_df_current_central_city], ignore_index=True)
        
    return metrics_df_central_cities

def apply_ml_models_for_bordering_cities(clusters, neural_network_models):
    
    metrics_df_bordering_cities = None
    
    for cluster_name, cities_dict in clusters.items():
        print(f'Model {cluster_name}:')
        MODEL = neural_network_models[cluster_name]
        
        bordering_cities = list(cities_dict.keys())
        bordering_cities.remove(cluster_name)
        
        for city in bordering_cities:
            print(f'\tCity {city}')
            DATASET = clusters[cluster_name][city]
            
            _ , metrics_df_bordering_cities_current_model = MODEL.use_neural_network(dataset=DATASET)
    
        # Run once for every central city, not for every bordering city:
        if metrics_df_bordering_cities is None:
            metrics_df_bordering_cities = metrics_df_bordering_cities_current_model
        else:
            metrics_df_bordering_cities = pd.concat([metrics_df_bordering_cities, metrics_df_bordering_cities_current_model], ignore_index=True)
    
    return metrics_df_bordering_cities

def _fmt_table(headers, rows):
    """Format a list of rows into a fixed-width ASCII table with pipe borders."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt_row(row):
        parts = []
        for i, cell in enumerate(row):
            s = str(cell)
            # Left-align first column (city names), right-align numeric columns
            parts.append(s.ljust(widths[i]) if i == 0 else s.rjust(widths[i]))
        return '| ' + ' | '.join(parts) + ' |'

    divider = '+-' + '-+-'.join('-' * w for w in widths) + '-+'
    lines = [divider, fmt_row(headers), divider]
    for row in rows:
        lines.append(fmt_row(row))
    lines.append(divider)
    return '\n'.join(lines)

def _fmt_float(value):
    """Format a float for table display, handling NaN."""
    try:
        if pd.isna(value):
            return 'N/A'
        return f'{value:.4f}'
    except (TypeError, ValueError):
        return 'N/A'

def generate_results_report(metrics_df_central, metrics_df_bordering):
    """
    Generate a text report with comparison tables and conclusions.
    Saves to OUTPUT_DIR_ADDR/ml_model_results.txt.
    Expects both DataFrames to still contain the 'Agrupamento' column.
    """
    lines = []

    WIDTH = 70
    def section(title):
        lines.append('')
        lines.append('=' * WIDTH)
        lines.append(f'  {title}')
        lines.append('=' * WIDTH)

    lines.append('=' * WIDTH)
    lines.append(' ' * ((WIDTH - len('ML MODEL RESULTS REPORT')) // 2) + 'ML MODEL RESULTS REPORT')
    lines.append('=' * WIDTH)

    # ------------------------------------------------------------------
    # Section 1: Central city own-data performance (train 80% vs test 20%)
    # ------------------------------------------------------------------
    section('1. CENTRAL CITY PERFORMANCE  (model trained and tested on its own data)')
    lines.append('')
    lines.append('Metrics: NumPy implementation  |  Train = 80% split  |  Test = 20% split')
    lines.append('')

    headers = ['City (Model)', 'Cluster',
               'MAE Train', 'RMSE Train', 'R² Train',
               'MAE Test',  'RMSE Test',  'R² Test']

    rows = []
    for _, row in metrics_df_central.iterrows():
        rows.append([
            row['Municipio Previsto'],
            row['Agrupamento'],
            _fmt_float(row['MAE 80% Numpy']),
            _fmt_float(row['RMSE 80% Numpy']),
            _fmt_float(row['R^2 80% Numpy']),
            _fmt_float(row['MAE 20% Numpy']),
            _fmt_float(row['RMSE 20% Numpy']),
            _fmt_float(row['R^2 20% Numpy']),
        ])

    lines.append(_fmt_table(headers, rows))

    # ------------------------------------------------------------------
    # Section 2: Bordering city transfer performance (test 20%)
    # ------------------------------------------------------------------
    section('2. BORDERING CITY PERFORMANCE  (model applied to neighbouring cities)')
    lines.append('')
    lines.append('Metrics: NumPy implementation  |  Test = 20% split')
    lines.append('')

    headers_b = ['City Predicted', 'Model Trained On',
                 'MAE Test', 'RMSE Test', 'R² Test']

    rows_b = []
    for _, row in metrics_df_bordering.iterrows():
        rows_b.append([
            row['Municipio Previsto'],
            row['Municipio Treinado'],
            _fmt_float(row['MAE 20% Numpy']),
            _fmt_float(row['RMSE 20% Numpy']),
            _fmt_float(row['R^2 20% Numpy']),
        ])

    lines.append(_fmt_table(headers_b, rows_b))

    # ------------------------------------------------------------------
    # Section 3: Model ranking by test R²
    # ------------------------------------------------------------------
    section('3. MODEL RANKING  (central city models ranked by test R²)')
    lines.append('')

    ranking = metrics_df_central[['Municipio Previsto', 'R^2 20% Numpy', 'MAE 20% Numpy', 'RMSE 20% Numpy']].copy()
    ranking = ranking.sort_values('R^2 20% Numpy', ascending=False).reset_index(drop=True)

    rows_r = []
    for rank, (_, row) in enumerate(ranking.iterrows(), start=1):
        rows_r.append([
            str(rank),
            row['Municipio Previsto'],
            _fmt_float(row['R^2 20% Numpy']),
            _fmt_float(row['MAE 20% Numpy']),
            _fmt_float(row['RMSE 20% Numpy']),
        ])

    lines.append(_fmt_table(['Rank', 'City (Model)', 'R² Test', 'MAE Test', 'RMSE Test'], rows_r))

    # ------------------------------------------------------------------
    # Section 4: Overfitting analysis (train vs test gap)
    # ------------------------------------------------------------------
    section('4. OVERFITTING ANALYSIS  (train R² minus test R² per model)')
    lines.append('')
    lines.append('A large positive gap suggests the model memorised training data.')
    lines.append('')

    rows_o = []
    for _, row in metrics_df_central.iterrows():
        r2_train = row['R^2 80% Numpy']
        r2_test  = row['R^2 20% Numpy']
        try:
            gap = r2_train - r2_test if (not pd.isna(r2_train) and not pd.isna(r2_test)) else float('nan')
        except TypeError:
            gap = float('nan')
        flag = ''
        if not pd.isna(gap):
            if gap > 0.2:
                flag = '  *** HIGH'
            elif gap > 0.1:
                flag = '  *  MODERATE'
        rows_o.append([
            row['Municipio Previsto'],
            _fmt_float(r2_train),
            _fmt_float(r2_test),
            _fmt_float(gap) + flag,
        ])

    rows_o.sort(key=lambda r: float(r[3].split()[0]) if r[3] != 'N/A' else -999, reverse=True)
    lines.append(_fmt_table(['City (Model)', 'R² Train', 'R² Test', 'Gap (Train − Test)'], rows_o))

    # ------------------------------------------------------------------
    # Section 5: Generalization analysis (central test vs bordering test)
    # ------------------------------------------------------------------
    section('5. GENERALIZATION ANALYSIS  (central model test R² vs. bordering city test R²)')
    lines.append('')
    lines.append('Shows how well each model transfers beyond its training city.')
    lines.append('')

    central_lookup = metrics_df_central.set_index('Municipio Treinado')

    rows_g = []
    for _, row in metrics_df_bordering.iterrows():
        trained_on = row['Municipio Treinado']
        r2_central = central_lookup.loc[trained_on, 'R^2 20% Numpy'] if trained_on in central_lookup.index else float('nan')
        r2_border  = row['R^2 20% Numpy']
        try:
            drop = r2_central - r2_border if (not pd.isna(r2_central) and not pd.isna(r2_border)) else float('nan')
        except TypeError:
            drop = float('nan')
        rows_g.append([
            row['Municipio Previsto'],
            trained_on,
            _fmt_float(r2_central),
            _fmt_float(r2_border),
            _fmt_float(drop),
        ])

    rows_g.sort(key=lambda r: float(r[4]) if r[4] != 'N/A' else -999, reverse=True)
    lines.append(_fmt_table(
        ['City Predicted', 'Model', 'R² Central (test)', 'R² Bordering (test)', 'Drop'],
        rows_g))

    # ------------------------------------------------------------------
    # Section 6: Conclusions
    # ------------------------------------------------------------------
    section('6. CONCLUSIONS')
    lines.append('')

    r2_col = 'R^2 20% Numpy'
    mae_col = 'MAE 20% Numpy'
    rmse_col = 'RMSE 20% Numpy'

    valid_central = metrics_df_central.dropna(subset=[r2_col])

    if not valid_central.empty:
        best_row  = valid_central.loc[valid_central[r2_col].idxmax()]
        worst_row = valid_central.loc[valid_central[r2_col].idxmin()]
        avg_r2   = valid_central[r2_col].mean()
        avg_mae  = valid_central[mae_col].mean()
        avg_rmse = valid_central[rmse_col].mean()

        lines.append(f'  Best central-city model  : {best_row["Municipio Previsto"]}')
        lines.append(f'    R² = {_fmt_float(best_row[r2_col])}  |  MAE = {_fmt_float(best_row[mae_col])}  |  RMSE = {_fmt_float(best_row[rmse_col])}')
        lines.append('')
        lines.append(f'  Worst central-city model : {worst_row["Municipio Previsto"]}')
        lines.append(f'    R² = {_fmt_float(worst_row[r2_col])}  |  MAE = {_fmt_float(worst_row[mae_col])}  |  RMSE = {_fmt_float(worst_row[rmse_col])}')
        lines.append('')
        lines.append(f'  Average across all central-city models (test split):')
        lines.append(f'    R² = {_fmt_float(avg_r2)}  |  MAE = {_fmt_float(avg_mae)}  |  RMSE = {_fmt_float(avg_rmse)}')
        lines.append('')

        # Overfitting summary
        metrics_df_central_copy = metrics_df_central.copy()
        metrics_df_central_copy['gap'] = metrics_df_central_copy['R^2 80% Numpy'] - metrics_df_central_copy['R^2 20% Numpy']
        high_overfit = metrics_df_central_copy[metrics_df_central_copy['gap'] > 0.2]['Municipio Previsto'].tolist()
        mod_overfit  = metrics_df_central_copy[(metrics_df_central_copy['gap'] > 0.1) &
                                               (metrics_df_central_copy['gap'] <= 0.2)]['Municipio Previsto'].tolist()

        if high_overfit:
            lines.append(f'  High overfitting (gap > 0.2) detected in: {", ".join(high_overfit)}')
        if mod_overfit:
            lines.append(f'  Moderate overfitting (gap > 0.1) detected in: {", ".join(mod_overfit)}')
        if not high_overfit and not mod_overfit:
            lines.append('  No significant overfitting detected across central-city models.')
        lines.append('')

    valid_bordering = metrics_df_bordering.dropna(subset=[r2_col])

    if not valid_bordering.empty:
        avg_r2_b   = valid_bordering[r2_col].mean()
        avg_mae_b  = valid_bordering[mae_col].mean()
        avg_rmse_b = valid_bordering[rmse_col].mean()

        lines.append('  Bordering-city transfer performance (average across all bordering cities):')
        lines.append(f'    R² = {_fmt_float(avg_r2_b)}  |  MAE = {_fmt_float(avg_mae_b)}  |  RMSE = {_fmt_float(avg_rmse_b)}')
        lines.append('')

        best_b  = valid_bordering.loc[valid_bordering[r2_col].idxmax()]
        worst_b = valid_bordering.loc[valid_bordering[r2_col].idxmin()]

        lines.append(f'  Best bordering-city prediction : {best_b["Municipio Previsto"]} (model: {best_b["Municipio Treinado"]})')
        lines.append(f'    R² = {_fmt_float(best_b[r2_col])}')
        lines.append(f'  Worst bordering-city prediction : {worst_b["Municipio Previsto"]} (model: {worst_b["Municipio Treinado"]})')
        lines.append(f'    R² = {_fmt_float(worst_b[r2_col])}')
        lines.append('')

        if not valid_central.empty:
            central_mean_r2   = valid_central[r2_col].mean()
            bordering_mean_r2 = valid_bordering[r2_col].mean()
            diff = central_mean_r2 - bordering_mean_r2
            lines.append(f'  Average R² drop when applying model to bordering cities: {_fmt_float(diff)}')
            if diff < 0.05:
                lines.append('  -> Models generalize well to neighbouring cities.')
            elif diff < 0.15:
                lines.append('  -> Models show moderate performance drop on neighbouring cities.')
            else:
                lines.append('  -> Models show significant performance drop on neighbouring cities;')
                lines.append('     consider training on a larger or more representative area.')

    lines.append('')
    lines.append('=' * WIDTH)
    lines.append('')

    report_text = '\n'.join(lines)
    report_path = f'{OUTPUT_DIR_ADDR}/ml_model_results.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f'\tSaved results report to {report_path}')

def save_ml_models_for_later_reuse(neural_network_models):
    if os.path.isdir(f'{OUTPUT_DIR_ADDR}/Models'):
        shutil.rmtree(f'{OUTPUT_DIR_ADDR}/Models')
    os.makedirs(f'{OUTPUT_DIR_ADDR}/Models')
    
    for name, model_object in neural_network_models.items():
        model_object.model.save        (f'{OUTPUT_DIR_ADDR}/Models/{name}.keras'     )
        model_object.model.save_weights(f'{OUTPUT_DIR_ADDR}/Models/{name}.weights.h5')

def save_results(metrics_df_all_bordering_cities, metrics_df_central_cities_only, neural_network_models):
    # Sort by cluster name first, then by city name
    metrics_df_all_bordering_cities = metrics_df_all_bordering_cities.sort_values(
        by=['Agrupamento', 'Municipio Previsto'], ignore_index=True)
    metrics_df_central_cities_only = metrics_df_central_cities_only.sort_values(
        by=['Agrupamento', 'Municipio Previsto'], ignore_index=True)
    
    generate_results_report(metrics_df_central_cities_only, metrics_df_all_bordering_cities)

    metrics_df_all_bordering_cities = metrics_df_all_bordering_cities.drop('Agrupamento', axis='columns')
    metrics_df_central_cities_only  = metrics_df_central_cities_only .drop('Agrupamento', axis='columns')

    metrics_df_all_bordering_cities.to_excel(f'{OUTPUT_DIR_ADDR}/metrics_bordering_cities.xlsx', index=False)
    metrics_df_central_cities_only .to_excel(f'{OUTPUT_DIR_ADDR}/metrics_central_cities.xlsx'  , index=False)

    save_ml_models_for_later_reuse(neural_network_models)

print('PREPARATION: START')
THE_PLOTTER = Plotter()

clusters = InputDataLoader(INPUT_DIR_ADDR).get_cluster_memberships()
print('\tLoaded all datasets')

make_output_dirs(OUTPUT_DIR_ADDR, clusters)
print('\tMade output directories for all cities')
print('PREPARATION: END')

print('CREATION: START')
neural_network_models = instantiate_ml_models_for_central_cities()
print('CREATION: END')

print('TRAINING: START')
metrics_df_central_cities_only = train_ml_models_for_central_cities()
print('TRAINING: END')

print('APPLYING: START')
metrics_df_all_bordering_cities = apply_ml_models_for_bordering_cities(clusters, neural_network_models)
print('APPLYING: END')

print('TERMINATION: START')
save_results(metrics_df_all_bordering_cities, metrics_df_central_cities_only, neural_network_models)
print('TERMINATION: END')