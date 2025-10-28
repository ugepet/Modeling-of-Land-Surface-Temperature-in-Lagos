#!/usr/bin/env python3
"""
biophysical_model_for_lagos_canopy-temperature_CLEANED.py
Reproducible script for training Random Forest Models 1-3 to predict LST in Lagos.
Requires: Lagos_full_stack_100m.tif placed in the same directory.
Outputs saved to ./outputs/
"""

import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay
from joblib import dump

# ---------------------- User parameters ----------------------
stack_path = "Lagos_full_stack_100m.tif"  # multiband stack containing predictors and LST
output_csv = "Lagos_RF_input_data.csv"
outputs_dir = "outputs"
os.makedirs(outputs_dir, exist_ok=True)

# Band mapping (confirmed)
band_map = {
    1: 'lst_l8',
    2: 'lst_modis',
    3: 'ndvi',
    4: 'canopy',
    5: 'built_pct',
    6: 'pop_100m',
    7: 'elevation',
    8: 'ndbi',
    9: 'lulc',
    10: 'albedo'
}

# Predictors for Model 3
predictors_model3 = ['ndvi', 'canopy', 'built_pct', 'elevation', 'ndbi', 'lulc', 'albedo']
target = 'lst_l8'

# Random Forest parameters (consistent across models)
rf_params = {
    'n_estimators': 500,
    'max_depth': 25,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'bootstrap': True,
    'random_state': 42,
    'n_jobs': -1
}

# ---------------------- 1. Read raster and extract stack to CSV ----------------------
def raster_to_dataframe(stack_path, band_map, max_missing=3):
    with rasterio.open(stack_path) as src:
        stack = src.read()  # (bands, rows, cols)
        transform = src.transform
        height = src.height
        width = src.width
        count = src.count
        print(f"Loaded raster with {count} bands, size {width}x{height}")

    data = {}
    for i in range(stack.shape[0]):
        bname = band_map.get(i+1, f'band_{i+1}')
        arr = stack[i].astype(float)
        data[bname] = arr.flatten()

    # coordinates
    rows_idx, cols_idx = np.indices((height, width))
    xs, ys = rasterio.transform.xy(transform, rows_idx, cols_idx)
    xs = np.array(xs).flatten()
    ys = np.array(ys).flatten()
    data['lon'] = xs[:len(data[next(iter(data))])]
    data['lat'] = ys[:len(data[next(iter(data))])]

    df = pd.DataFrame(data)
    print("Initial flattened dataframe shape:", df.shape)

    # Replace common NoData values and infinities with NaN
    df = df.replace([-9999, 0, np.inf, -np.inf], np.nan)

    # Keep rows with at most max_missing missing cells
    df = df[df.isna().sum(axis=1) <= max_missing]

    # Drop rows missing the target
    df = df.dropna(subset=[target])
    print("Dataframe shape after cleaning:", df.shape)
    return df

if not os.path.exists(output_csv):
    print("Extracting raster bands and creating CSV...")
    df = raster_to_dataframe(stack_path, band_map)
    cols_to_save = list(band_map.values()) + ['lon', 'lat']
    # ensure target and predictors are in csv
    available_cols = [c for c in cols_to_save if c in df.columns]
    df[available_cols].to_csv(output_csv, index=False)
    print("Saved cleaned CSV to", output_csv)
else:
    print("Loading existing CSV:", output_csv)
    df = pd.read_csv(output_csv)

# ---------------------- 2. Prepare data for modeling ----------------------
# Select predictors and target; allow that some predictors might be missing in CSV
available_predictors = [p for p in predictors_model3 if p in df.columns]
print("Available predictors:", available_predictors)

# Drop rows missing target
df = df.dropna(subset=[target])
X = df[available_predictors].copy()
y = df[target].copy()

# Impute missing predictor values with column mean
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# ---------------------- 3. Train Models 1-3 ----------------------
model_defs = {
    'Model1': [p for p in ['ndvi', 'canopy'] if p in X.columns],
    'Model2': [p for p in ['ndvi', 'canopy', 'built_pct', 'elevation'] if p in X.columns],
    'Model3': available_predictors
}

models = {}
metrics = {}

for mname, feats in model_defs.items():
    print(f"\nTraining {mname} with features: {feats}")
    rf = RandomForestRegressor(**rf_params)
    rf.fit(X_train[feats], y_train)
    models[mname] = rf
    # Evaluate
    y_pred = rf.predict(X_test[feats])
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    metrics[mname] = {'r2': float(r2), 'rmse': float(rmse), 'mae': float(mae)}
    print(f"{mname} -- R2: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    # save model
    dump(rf, os.path.join(outputs_dir, f"{mname}_RF_model.joblib"))

# Save metrics
metrics_df = pd.DataFrame(metrics).T.reset_index().rename(columns={'index':'model'})
metrics_df.to_csv(os.path.join(outputs_dir, 'model_performance_summary.csv'), index=False)
print("Saved model_performance_summary.csv in outputs/")

# ---------------------- 4. Feature importance & PDPs for Model3 ----------------------
if 'Model3' in models:
    model3 = models['Model3']
    feat_names = model_defs['Model3']

    # Feature importance
    importances = model3.feature_importances_
    feat_imp = pd.Series(importances, index=feat_names).sort_values(ascending=False)
    plt.figure(figsize=(8,5))
    sns.barplot(x=feat_imp.values, y=feat_imp.index)
    plt.title('Feature Importance (Model 3)')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, 'feature_importance_model3.png'), dpi=300)
    plt.close()

    # Partial Dependence Plots for selected features (if present)
    pdp_features = [f for f in ['ndvi', 'built_pct', 'ndbi', 'albedo'] if f in feat_names]
    if len(pdp_features) > 0:
        try:
            fig, ax = plt.subplots(2,2, figsize=(12,8))
            PartialDependenceDisplay.from_estimator(model3, X_imputed, pdp_features, kind='average', grid_resolution=100, ax=ax)
            plt.suptitle('Partial Dependence Plots (Model 3)')
            plt.tight_layout(rect=[0,0,1,0.96])
            plt.savefig(os.path.join(outputs_dir, 'PDP_composite_model3.png'), dpi=300)
            plt.close()
        except Exception as e:
            print("Warning: PDP generation failed:", e)

# ---------------------- 5. Spatial prediction with Model3 ----------------------
def predict_raster_model3(stack_path, model, feature_list, band_map, outputs_dir, out_name='Predicted_LST_Model3_100m.tif'):
    with rasterio.open(stack_path) as src:
        meta = src.meta.copy()
        meta.update(count=1, dtype='float32')
        rows = src.height
        cols = src.width

        # map feature name to band index
        band_indices = {v:k for k,v in band_map.items()}
        feat_band_idxs = [band_indices[f] for f in feature_list if f in band_indices]
        if len(feat_band_idxs) != len(feature_list):
            print("Warning: Not all feature bands found in raster. Skipping raster prediction.")
            return None

        pred_arr = np.full((rows, cols), np.nan, dtype='float32')
        window_size = 512
        for i in range(0, rows, window_size):
            for j in range(0, cols, window_size):
                h = min(window_size, rows - i)
                w = min(window_size, cols - j)
                window = Window(j, i, w, h)
                data_block = src.read(feat_band_idxs, window=window).astype(float)
                n_bands = data_block.shape[0]
                flat_block = data_block.reshape(n_bands, -1).T
                df_block = pd.DataFrame(flat_block, columns=feature_list)
                df_block = df_block.replace([-9999, np.inf, -np.inf], np.nan)
                df_block = df_block.fillna(X_imputed.mean())
                try:
                    preds = model.predict(df_block)
                except Exception as e:
                    print("Prediction failed on block:", e)
                    preds = np.full((h*w,), np.nan)
                preds = preds.reshape(h, w)
                pred_arr[i:i+h, j:j+w] = preds.astype('float32')

        out_pred_path = os.path.join(outputs_dir, out_name)
        meta.update(dtype='float32')
        with rasterio.open(out_pred_path, 'w', **meta) as dst:
            dst.write(pred_arr, 1)
        print("Saved predicted raster to", out_pred_path)
        return out_pred_path

# Run raster prediction if model3 available
if 'Model3' in models and len(model_defs['Model3'])>0:
    predict_raster_model3(stack_path, models['Model3'], model_defs['Model3'], band_map, outputs_dir)

# ---------------------- 6. Observed vs Predicted scatter and metrics for Model3 ----------------------
if 'Model3' in models:
    model3 = models['Model3']
    feats = model_defs['Model3']
    y_pred_test = model3.predict(X_test[feats])
    r2 = r2_score(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae = mean_absolute_error(y_test, y_pred_test)
    print(f"Model 3 test metrics: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred_test, s=8, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Observed lst_l8 (°C)')
    plt.ylabel('Predicted lst_l8 (°C)')
    plt.title(f'Observed vs Predicted (Model 3) R²={r2:.3f}')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, 'Scatter_Model3_TestSet.png'), dpi=300)
    plt.close()

# ---------------------- 7. LST by LULC boxplot ----------------------
try:
    if 'lulc' in df.columns and 'lst_l8' in df.columns:
        df_box = df.dropna(subset=['lulc','lst_l8']).copy()
        # simple mapping (update as needed)
        def classify_lulc(val):
            try:
                v = int(val)
            except:
                return 'Other'
            if v <= 3:
                return 'Water'
            elif v <=6:
                return 'Wetlands'
            elif v <=10:
                return 'Vegetation'
            elif v <=14:
                return 'Bare/Transitional'
            elif v <=20:
                return 'Built-up'
            else:
                return 'Other'
        df_box['LULC_Class'] = df_box['lulc'].apply(classify_lulc)
        order = ['Water','Wetlands','Vegetation','Bare/Transitional','Built-up']
        plt.figure(figsize=(8,6))
        sns.boxplot(x='LULC_Class', y='lst_l8', data=df_box, order=order)
        plt.xlabel('LULC Class')
        plt.ylabel('lst_l8 (°C)')
        plt.title('Distribution of lst_l8 across LULC classes')
        plt.tight_layout()
        plt.savefig(os.path.join(outputs_dir, 'LST_by_LULC_Boxplot.png'), dpi=300)
        plt.close()
except Exception as e:
    print("Boxplot generation skipped:", e)

# ---------------------- 8. Correlation matrix ----------------------
try:
    df_sub = df[[c for c in predictors_model3 if c in df.columns] + [target]].dropna()
    corr = df_sub.corr()
    plt.figure(figsize=(9,7))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5, square=True, cbar_kws={'label':'Pearson r'})
    plt.title('Correlation Matrix of Predictors and lst_l8')
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, 'Correlation_Matrix_lst_l8.png'), dpi=300)
    plt.close()
except Exception as e:
    print("Correlation matrix generation skipped:", e)

# ---------------------- 9. Save artifacts summary ----------------------
try:
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.to_csv(os.path.join(outputs_dir, 'model_metrics_table.csv'))
    if 'feat_imp' in globals():
        feat_imp.to_csv(os.path.join(outputs_dir, 'feature_importance_model3.csv'))
    print("Saved artifacts summary to outputs/")
except Exception as e:
    print("Saving artifacts failed:", e)

print("Script finished. Check the outputs/ folder for results.")
