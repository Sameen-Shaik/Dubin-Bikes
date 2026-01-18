# Dublin Bikes Data Analysis - Strategic Availability Forecasting

A comprehensive machine learning project that predicts bike availability across Dublin Bikes stations using historical data.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Analysis Pipeline](#analysis-pipeline)
- [Models & Results](#models--results)
- [Visualizations](#visualizations)
- [Key Findings](#key-findings)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)

## Project Overview

This project aims to predict the number of available bikes at Dublin Bikes stations using machine learning techniques. The analysis includes:

- **Data Cleaning & Preprocessing**: Handling missing values, duplicates, and logical inconsistencies
- **Exploratory Data Analysis**: Visualizing temporal and spatial patterns in bike usage
- **Feature Engineering**: Creating cyclical time features and geographic encodings
- **Model Development**: Training and comparing multiple regression models
- **Feature Selection**: Identifying the most predictive attributes using importance ranking and RFE

### Objective

Predict `AVAILABLE_BIKES` - the number of bikes currently available for rental at a given station - to optimize redistribution logistics and improve service availability.

## Dataset

**Source**: [Dublin Bikes API - Smart Dublin](https://data.smartdublin.ie/dataset/dublinbikes-api)

### Dataset Characteristics

| Attribute | Description |
|-----------|-------------|
| **Records** | 163,590 rows |
| **Features** | 11 columns |
| **Time Period** | June 2023 |
| **Target Variable** | AVAILABLE_BIKES |

### Features

| Column | Type | Description |
|--------|------|-------------|
| STATION ID | Integer | Unique identifier for each station |
| TIME | Datetime | Timestamp of the record |
| LAST UPDATED | Datetime | Last update time from sensor |
| NAME | String | Station name |
| BIKE_STANDS | Integer | Total bike capacity at station |
| AVAILABLE_BIKE_STANDS | Integer | Empty stands available |
| AVAILABLE_BIKES | Integer | Bikes available for rent |
| STATUS | String | Station status (OPEN/CLOSED) |
| ADDRESS | String | Station address |
| LATITUDE | Float | Geographic latitude |
| LONGITUDE | Float | Geographic longitude |

## Project Structure

```
Dublin Bikes Data Analysis/
├── Data/
│   ├── dataset.csv                    # Raw dataset
│   ├── dataset_cleaned.csv            # Cleaned dataset
│   └── dataset_preprocessed.csv       # Preprocessed dataset ready for modeling
├── Outputs/
│   ├── Figures/
│   │   ├── Availablity of Bikes.png
│   │   ├── Availablity of Bikes - Weekend Vs Weekday.png
│   │   ├── Mean & Median Availablity of Bikes.png
│   │   └── Bikes Availability Heatmap.png
│   └── Models/
│       ├── XG-Boost Model.joblib
│       └── MLP Model.joblib
├── Result Datasets/
│
├── Notebook 1 -- Data cleaning.ipynb      # Data cleaning and preparation
├── Notebook 2 -- Data Visualization.ipynb # Exploratory data analysis
├── Notebook 3 -- Model Creation.ipynb     # Model training and evaluation
├── Notebook 4 -- Feature Selection.ipynb  # Feature importance and RFE
├── Script -- Training pipeline.py         # Complete training pipeline script
└── README.md                              # This file
```

## Installation

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab

### Dependencies

```bash
pip install -r requirements.txt
```
or

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib tqdm
```

### Required Libraries

```python
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.12.0
scikit-learn>=1.0.0
xgboost>=1.6.0
joblib>=1.1.0
tqdm>=4.64.0
```

## Usage

### Option 1: Run Individual Notebooks

Execute notebooks in sequence:

1. **Data Cleaning**: `Notebook 1 -- Data cleaning.ipynb`
2. **Visualization**: `Notebook 2 -- Data Visualization.ipynb`
3. **Model Training**: `Notebook 3 -- Model Creation.ipynb`
4. **Feature Selection**: `Notebook 4 -- Feature Selection.ipynb`

### Option 2: Run Complete Pipeline

Execute the complete training pipeline:

```bash
python "Script -- Training pipeline.py"
```

### Loading Pre-trained Models

```python
import joblib

# Load XGBoost model
xgb_model = joblib.load('Outputs/Models/XG-Boost Model.joblib')

# Load MLP model
mlp_model = joblib.load('Outputs/Models/MLP Model.joblib')

# Make predictions
predictions = xgb_model.predict(X_test)
```

## Analysis Pipeline

### 1. Data Cleaning

- **Missing Values**: Handled using dropna() for sparse nulls
- **Duplicates**: Removed duplicate records
- **Business Rules Validation**:
  - Rule 1: `AVAILABLE_BIKES >= 0` (no negative bikes)
  - Rule 2: `AVAILABLE_BIKES <= BIKE_STANDS` (bikes cannot exceed capacity)
  - Rule 3: `AVAILABLE_BIKE_STANDS <= BIKE_STANDS` (stands cannot exceed capacity)

### 2. Feature Engineering

#### Temporal Features
- Hour, Minute, Day of Week, Month, IS_WEEKEND flag

#### Cyclical Encoding
Time features encoded using sine/cosine transformations to preserve cyclical nature:
```python
df['HOUR_SIN'] = np.sin(2 * np.pi * df['HOUR'] / 24)
df['HOUR_COS'] = np.cos(2 * np.pi * df['HOUR'] / 24)
```

#### Geographic Encoding
Latitude and Longitude encoded similarly:
```python
df['LAT_SIN'] = np.sin(df['LATITUDE'] * np.pi / 180)
df['LAT_COS'] = np.cos(df['LATITUDE'] * np.pi / 180)
```

### 3. Data Preprocessing

- **Label Encoding**: STATUS column (OPEN/CLOSED) encoded to numeric
- **Min-Max Scaling**: Numerical features normalized to [0, 1] range
- **Feature Selection**: Removed non-predictive columns (IDs, names, raw timestamps)

## Models & Results

### Model Performance Comparison

| Model | MAE | RMSE | R² Score | Training Time |
|-------|-----|------|----------|---------------|
| **XGBoost** | **4.81** | **6.21** | **0.61** | **1.24s** |
| Gradient Boosting | 5.50 | 6.90 | 0.52 | ~45s |
| Random Forest | 5.44 | 7.29 | 0.47 | 7.50s |
| MLP Neural Network | 8.06 | 9.68 | 0.05 | 38.24s |
| Baseline (Mean) | 8.26 | - | - | - |

### Best Model: XGBoost

```python
XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
```

### 10-Fold Cross-Validation Results

| Model | Mean CV MAE | Improvement vs Baseline |
|-------|-------------|------------------------|
| XGBoost | 5.42 | **34.39%** |
| MLP | 7.99 | 3.22% |

## Visualizations

### 1. Bikes Availability Throughout the Day
Line plot showing hourly average bike availability - reveals distinct peaks and troughs corresponding to commuting patterns.

### 2. Weekend vs Weekday Comparison
Comparison plot showing different usage patterns:
- **Weekdays**: Sharp variance indicative of commuter traffic
- **Weekends**: Flatter pattern suggesting leisure usage

### 3. Average/Median Bikes by Day of Week
Bar charts comparing mean and median availability across all days.

### 4. Availability Heatmap
Hour × Day of Week heatmap highlighting "hotspots" of high availability and "coldspots" of high demand.

## Key Findings

1. **XGBoost is the Superior Model**: Achieved the best balance of accuracy (MAE ≈ 4.8) and computational speed (~1.2 seconds)

2. **Neural Networks Underperformed**: MLPs struggled with this tabular dataset compared to tree-based ensemble methods

3. **Strong Temporal Patterns**: 
   - Clear commuter patterns during weekday mornings (8 AM) and evenings (5 PM)
   - Weekend usage shows different, more distributed patterns

4. **Feature Engineering Impact**: Cyclical encoding of time features significantly improved model performance

5. **Feature Selection Benefits**: Model complexity can be reduced significantly (10 features vs full set) without major accuracy penalty - valuable for deployment

### Most Important Features (XGBoost)

1. BIKE_STANDS (station capacity)
2. HOUR_SIN / HOUR_COS (time of day)
3. DOW_SIN / DOW_COS (day of week)
4. IS_WEEKEND
5. LAT_SIN / LON_SIN (location)

## Technologies Used

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.x |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Model Persistence** | Joblib |
| **Development** | Jupyter Notebook |

## Future Improvements

- Integrate weather data for improved predictions
- Add real-time streaming predictions
- Deploy as a web service API
- Include station-specific models for high-traffic locations
- Implement time-series models (LSTM, Prophet) for sequential forecasting

## License

MIT License