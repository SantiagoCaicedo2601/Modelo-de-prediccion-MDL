## Modelo-de-prediccion-MDL
En este repositorio se muestra un modelo de prediccion de contaminacion de gases de efecto invernadero basado en bases de datos suministradas de proyectos de mecanismo de desarrollo limpio 

### Importar librerías de python

	import numpy as np
	import pandas as pd
	from sklearn.preprocessing import MinMaxScaler
	from tensorflow.keras.models import Sequential
	from tensorflow.keras.layers import LSTM, Dense
	import matplotlib.pyplot as plt
	from scipy.interpolate; import make_interp_spline>

#### Función para cargar datos de Excel
    def load_data(file):
	      data = pd.read_excel(file)
          return data

#### Función para preparar los datos

    def prepare_data(data, time_steps):
          X, y = [], []
          for i in range(len(data) - time_steps):
               X.append(data.iloc[i:i + time_steps, 0:3])
               y.append(data.iloc[i + time_steps, 0:3])
           return np.array(X), np.array(y)>

### Parámetros
     file = 'Nombre del archivo'
     time_steps = 10
     num_years = 20
     num_simulations = 10

### Cargar los datos
    data = load_data(file)

### Normalizar los datos
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(data)

### Crear el dataset
    X, y = prepare_data(pd.DataFrame(data_normalized), time_steps)

### Dividir los datos en conjuntos de entrenamiento y prueba
    train_size = int(len(X) * 0.7)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

### Definir el modelo LSTM
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=False, input_shape=(time_steps, 3)))
    model.add(Dense(units=3))

### Compilar el modelo
    model.compile(optimizer='adam', loss='mean_squared_error')

### Entrenar el modelo
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Función para hacer predicciones futuras
    def make_future_predictions(model, current_data, num_years, noise_level):
           future_predictions = []
           for _ in range(num_years):
                next_year_prediction = model.predict(current_data)
                next_year_prediction += np.random.normal(0, noise_level, next_year_prediction.shape)
                future_predictions.append(next_year_prediction)
                current_data = np.concatenate([current_data[:, 1:, :], next_year_prediction.reshape(1, 1, 3)], axis=1)
           return np.array(future_predictions).reshape(-1, 3)

### Hacer múltiples simulaciones de predicciones futuras
    all_predictions = []
    for i in range(num_simulations):
         current_data = data_normalized[-time_steps:].reshape(1, time_steps, 3)
         noise_level = np.random.uniform(0.01, 0.1)
         future_predictions = make_future_predictions(model, current_data, num_years, noise_level)
         all_predictions.append(future_predictions)

### Desnormalizar las predicciones
     all_predictions_descaled = [scaler.inverse_transform(pred) for pred in all_predictions]

### Crear una lista de años
    years = np.array([2022 + i for i in range(num_years)])

### Graficar todas las simulaciones de predicciones
    plt.figure(figsize=(12, 8))

# Función para suavizar las líneas
    def smooth_line(x, y, factor=300):
    xnew = np.linspace(x.min(), x.max(), factor)
    spl = make_interp_spline(x, y, k=3)
    ynew = spl(xnew)
    return xnew, ynew

### Definir funciones para obtener colores en base a valores
    def get_green_color(value):
           return plt.cm.Greens((430 - abs(value - 430)) / 430)

    def get_yellow_color(value):
          return plt.cm.YlOrBr((value - 450) / 10)

    def get_red_color(value):
           return plt.cm.Reds((value - 475) / 10)

### Graficar las predicciones de cada simulación con colores dependiendo de los valores
    for i, prediction in enumerate(all_predictions_descaled):
         for j in range(3):
              x_smooth, y_smooth = smooth_line(years, prediction[:, j])
              for k in range(len(x_smooth) - 1):
                    value = y_smooth[k]
                    if value > 475:
                        color = get_red_color(value)
                    elif 430 <= value <= 475:
                         color = get_yellow_color(value)
                    else:
                         color = get_green_color(value)
                         plt.plot(x_smooth[k:k+2], y_smooth[k:k+2], linestyle='-', color=color, alpha=1)

### Nombre de ejes

    plt.xlabel('Año')
    plt.ylabel('Cantidad de kilotoneladas mitigadas de CO2')
    plt.title('Predicciones para los próximos años en Bucaramanga con 0.9')
    plt.grid(True)

### Ajustar la leyenda para evitar sobreposición
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1, 1), ncol=1, fontsize='small')

### Ajustar el rango del eje y
    plt.ylim(400, 520)

### Asegurarse de que el eje x muestre números enteros
    plt.xticks(years)
    plt.tight_layout()
    plt.show()

### Finalizacion y muestra de grafico

    print("Predicciones para los próximos 20 años (año por año):")
    for i, prediction in enumerate(all_predictions_descaled):
         print(f"Simulación {i+1}:")
         for j, year_prediction in enumerate(prediction):
              print(f"Año {2022+j}: {year_prediction}")
