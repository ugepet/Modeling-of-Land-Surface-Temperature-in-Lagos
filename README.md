# 🌍 Biophysical Modeling of Land Surface Temperature in Lagos Using Random Forest

This repository contains the full workflow and reproducible code for modeling **Land Surface Temperature (LST)** across **Lagos, Nigeria**, using **Machine Learning (Random Forest)** and **multisource remote sensing data**.  
The analysis forms part of the study:

> *“Modeling Urban Land Surface Temperature Using Machine Learning and Multisource Remote Sensing: A Case Study of Lagos, Nigeria.”*

---

## 🧠 Overview

Urban heat in Lagos is intensified by rapid urbanization, reduced vegetation, and increased impervious surfaces.  
This project integrates **biophysical, spectral, and structural predictors** to model and map LST at 100 m resolution.

The modeling framework was implemented using **Python (scikit-learn)** and **Random Forest regression**, with three progressively complex models:

| Model | Predictors Included | R² | RMSE (°C) | MAE (°C) |
|-------|---------------------|----|------------|-----------|
| Model 1 | NDVI, Canopy Height | 0.51 | 2.44 | 1.75 |
| Model 2 | NDVI, Canopy Height, Built_pct, Elevation | 0.66 | 2.04 | 1.43 |
| Model 3 | NDVI, Canopy Height, Built_pct, Elevation, NDBI, LULC, Albedo | **0.74** | **1.77** | **1.21** |

Model 3 was selected for final prediction, residual mapping, and interpretation.

---

## 🧩 Repository Structure

```
├── biophysical_model_for_lagos_canopy-temperature_CLEANED.ipynb   # Main Jupyter Notebook
├── biophysical_model_for_lagos_canopy-temperature_CLEANED.py      # Equivalent Python script
├── Lagos_full_stack_100m.tif                                      # Input multiband raster stack (user-supplied)
├── outputs/                                                       # All model outputs
│   ├── Predicted_LST_Model3_100m.tif
│   ├── Residuals_Model3_100m.tif
│   ├── feature_importance_model3.png
│   ├── PDP_composite_model3.png
│   ├── Scatter_Model3_TestSet.png
│   ├── LST_by_LULC_Boxplot.png
│   └── model_performance_summary.csv
└── README.md
```

---

## ⚙️ Dependencies

Install dependencies (Python 3.9+ recommended):

```bash
pip install numpy pandas rasterio geopandas shapely scikit-learn seaborn matplotlib joblib
```

---

## 🚀 How to Run

1. **Clone or Download** this repository.  
2. Ensure your input raster stack (`Lagos_full_stack_100m.tif`) is in the root folder.  
3. Open and run the notebook step-by-step:

   ```bash
   jupyter notebook biophysical_model_for_lagos_canopy-temperature_CLEANED.ipynb
   ```

   or run the `.py` script version:

   ```bash
   python biophysical_model_for_lagos_canopy-temperature_CLEANED.py
   ```

4. All outputs (figures, rasters, CSVs) will be saved automatically in the `outputs/` folder.

---

## 🛰️ Data Description

The raster stack (`Lagos_full_stack_100m.tif`) contains 10 bands:

| Band | Variable | Description |
|------|-----------|-------------|
| 1 | `lst_l8` | Land Surface Temperature (Landsat 8) |
| 2 | `lst_modis` | MODIS-derived LST (optional reference) |
| 3 | `ndvi` | Normalized Difference Vegetation Index |
| 4 | `canopy` | Vegetation / Tree Canopy Height |
| 5 | `built_pct` | Built-up percentage (imperviousness) |
| 6 | `pop_100m` | Population density (persons / 100 m grid) |
| 7 | `elevation` | Surface elevation (SRTM DEM) |
| 8 | `ndbi` | Normalized Difference Built-up Index |
| 9 | `lulc` | Land Use / Land Cover class |
| 10 | `albedo` | Surface albedo |

All data were resampled to **100 m resolution** and reprojected to **WGS 84 / UTM Zone 31N**.

---

## 📈 Outputs

The following results are generated:
- Predicted LST raster (`Predicted_LST_Model3_100m.tif`)
- Residual raster (`Residuals_Model3_100m.tif`)
- Feature importance and PDPs (key drivers of heat)
- Observed vs. Predicted scatter plot (validation)
- Boxplot of LST across LULC classes
- Correlation matrix for all predictors

---

## 🧮 Model Reproducibility

All Random Forest models used the following configuration:

```python
RandomForestRegressor(
    n_estimators=500,
    max_depth=25,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)
```
Model 3 achieved **R² = 0.74**, **RMSE = 1.77°C**, **MAE = 1.21°C** on test data.

---

## 📚 Citation

If you use this workflow, please cite:

> Ugege, P. (2025). *Modeling Urban Land Surface Temperature Using Machine Learning and Multisource Remote Sensing: A Case Study of Lagos, Nigeria.* (Manuscript under review).

---

## 🌱 License

This project is shared under the **Creative Commons Attribution 4.0 International License (CC-BY 4.0)**.  
You are free to reuse and adapt it with proper attribution.
