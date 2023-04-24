import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from src.utils import data_loader, get_data_into_submission_format
from src.drop_columns import ColumnDropperTransformer
from src.ProcessData import ProcessingData
from sklearn.model_selection import GridSearchCV

# %% Load the data
df_train, df_test, df_train_features, df_label = data_loader()

# %% Tag the data with TRAIN and TEST
df_train_features.loc[:, "type"] = "TRAIN"
df_test.loc[:, "type"] = "TEST"

# %% Concat the data
df_total_no_label = pd.concat([df_train_features, df_test], axis=0)

# Turn date column into month/day
df_total_no_label.loc[:, "month"] = pd.to_datetime(df_total_no_label.loc[:, "week_start_date"]).dt.month
df_total_no_label.drop(columns="week_start_date", inplace=True)

# %% Simple Data Cleaning: 
# 1. Drop duplicates
# 2. Fill missing values (with forward fill - method)
ProcessingData.duplicates_drop(df_train)
ProcessingData.fill_data(df_train, fillType ='ffill')

# Trim selected columns to remove outliers based on the test data
to_trim = ['ndvi_ne', 'ndvi_nw',
       'ndvi_se', 'ndvi_sw', 'precipitation_amt_mm', 'reanalysis_air_temp_k',
       'reanalysis_avg_temp_k', 'reanalysis_dew_point_temp_k',
       'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k',
       'reanalysis_precip_amt_kg_per_m2',
       'reanalysis_relative_humidity_percent', 'reanalysis_sat_precip_amt_mm',
       'reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k',
       'station_avg_temp_c', 'station_diur_temp_rng_c', 'station_max_temp_c',
       'station_min_temp_c', 'station_precip_mm']
ProcessingData.drop_outlier(df_train, df_test, to_trim)


# %% fed the data
y = df_train.loc[:, 'total_cases']

# Turn "sawtooth"-like features into cyclical "sine"-features
df_total_no_label = ProcessingData.cyclical_encoding(df_total_no_label, "weekofyear")
df_total_no_label = ProcessingData.cyclical_encoding(df_total_no_label, "month")

X = df_total_no_label.loc[df_total_no_label.loc[:,'type'] == 'TRAIN']
X_test = df_total_no_label.loc[df_total_no_label.loc[:,'type'] == 'TEST']


ProcessingData.fill_data(X, fillType ='ffill')
ProcessingData.fill_data(X_test, fillType ='ffill')

pipe = make_pipeline(
    ColumnDropperTransformer(["city"]),
    StandardScaler(),
    SimpleImputer(),
    RandomForestRegressor()
)

# %% Param Grid

param_grid = {
    "randomforestregressor__n_estimators": [100, 200],
    "randomforestregressor__min_samples_split": [2, 20, 50]
}

# %% Gridsearch

gscv = GridSearchCV(pipe, param_grid, scoring="neg_mean_absolute_error")

# %% Training two different models for the two cities
sj_x_train = X.query("city=='sj'")
sj_y_train = df_train.query("city=='sj'").loc[:, "total_cases"]
iq_x_train = X.query("city=='iq'")
iq_y_train = df_train.query("city=='iq'").loc[:, "total_cases"]
sj_x_test = X_test.query("city=='sj'")
iq_x_test = X_test.query("city=='iq'")

sj_x_train = sj_x_train.drop(columns='type')                                
sj_y_train = sj_y_train.drop(columns='type')
iq_x_train = iq_x_train.drop(columns='type')
iq_y_train = iq_y_train.drop(columns='type')
sj_x_test = sj_x_test.drop(columns='type')
iq_x_test = iq_x_test.drop(columns='type')


# Fit Model for San Juan
sj_model = gscv.fit(sj_x_train, sj_y_train)
sj_best_model = gscv.best_estimator_

# Fit Model for Iquitos
iq_model = gscv.fit(iq_x_train, iq_y_train)
iq_best_model = gscv.best_estimator_

# Make predictions
sj_predictions = sj_best_model.predict(sj_x_test)
iq_predictions = iq_best_model.predict(iq_x_test)

# Concat predictions and create submission file
total_predictions = list(sj_predictions) + list(iq_predictions)
get_data_into_submission_format(total_predictions)