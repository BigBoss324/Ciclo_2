import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import pickle

# Cargar datos
DATA_PATH = "C:/Users/Alexis Taha/Desktop/Notebooks"
FILE_BIKERPRO = 'SeoulBikeData.csv'
bikerpro = pd.read_csv(os.path.join(DATA_PATH, FILE_BIKERPRO), encoding="ISO-8859-1")

clean_columns = [
    x.lower().\
        replace("(°c)", '').\
        replace("(%)", '').\
        replace(" (m/s)", '').\
        replace(" (10m)", '').\
        replace(" (mj/m2)", '').\
        replace("(mm)", '').\
        replace(" (cm)", '').\
        replace(" ", '_')
    for x in bikerpro.columns
    ]

bikerpro.columns = clean_columns

# Limpieza y preprocesamiento básico
bikerpro['date'] = pd.to_datetime(bikerpro['date'], format='%d/%m/%Y')
bikerpro['month'] = bikerpro['date'].dt.month
bikerpro['is_weekend'] = bikerpro['date'].dt.weekday >= 5
bikerpro['hour'] = bikerpro['hour'].astype('category')

# Variables a considerar
weather_cols = ['temperature', 'humidity', 'wind_speed', 'visibility',
                'dew_point_temperature', 'solar_radiation', 'rainfall', 'snowfall']
categorical_cols = ['seasons', 'holiday', 'functioning_day', 'hour', 'month', 'is_weekend']

# División de los datos en entrenamiento y prueba
X = bikerpro.sort_values(['date', 'hour'])
X_train = X.iloc[: int(X.shape[0] * 0.8)].drop('rented_bike_count', axis=1)
y_train = X.iloc[: int(X.shape[0] * 0.8)]['rented_bike_count']
X_test = X.iloc[int(X.shape[0] * 0.8):].drop('rented_bike_count', axis=1)
y_test = X.iloc[int(X.shape[0] * 0.8):]['rented_bike_count']

# Preprocesamiento
numerical_pipe = Pipeline([
    ('yeo_johnson', PowerTransformer()),  # Transformación Yeo-Johnson
    ('scaler', StandardScaler()),         # Escalado
    ('selector', VarianceThreshold())     # Selección por umbral de varianza
])

categorical_pipe = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore'))  # One-hot encoding
])

preprocessor = ColumnTransformer([
    ('num', numerical_pipe, weather_cols),
    ('cat', categorical_pipe, categorical_cols)
])

# Creación y entrenamiento del modelo
model = Pipeline([
    ('preprocessor', preprocessor),
    ('knn', KNeighborsRegressor(n_neighbors=30))  # KNN con 30 vecinos
])

model.fit(X_train, y_train)

# Guardar el modelo entrenado
with open('model_fe_engineering_selection.pkl', 'wb') as file:
    pickle.dump(model, file)

# Código para cargar y usar el modelo
with open('model_fe_engineering_selection.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
    predictions = loaded_model.predict(X_test)    
    print(predictions)    


# Carga del modelo y predicciones
with open('model_fe_engineering_selection.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
    y_test_pred = loaded_model.predict(X_test)    

# Cálculo del MSE para el conjunto de prueba
error_test = mean_squared_error(y_test, y_test_pred)

# Imprimir los errores
print("Predicciones:", y_test_pred)
print("Error RMSE en test:", np.sqrt(error_test))  # Imprimir el RMSE
