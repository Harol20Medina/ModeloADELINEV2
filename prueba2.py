import numpy as np
import pandas as pd

# Parámetros
alpha = 0.3  # Tasa de aprendizaje
tolerancia = 1e-5  # Para determinar convergencia
X = np.array([[0, 0, 1],
              [0, 1, 0],
              [0, 1, 1],
              [1, 0, 0],
              [1, 0, 1],
              [1, 1, 0],
              [1, 1, 1],
              [0, 0, 0],
              [1, 0, 0],
              [1, 0, 1]])

# Función para predecir la salida
def calcular_salida(X, pesos):
    return np.dot(X, pesos)

# Función para convertir patrón binario a decimal (esto será la salida deseada)
def binario_a_decimal(binario):
    return binario[0] * 2**2 + binario[1] * 2**1 + binario[2] * 2**0

# Inicializar pesos
pesos = np.array([0.631, 0.840, 0.394])  # Pesos iniciales arbitrarios

# Inicializamos las columnas
data = {
    "Patrón Binario": [],
    "S.D. (z(k))": [],
    "Peso Antes (w1, w2, w3)": [],
    "Salida (S(k))": [],
    "Error (d - y)": [],
    "Ajuste de Pesos (Δw1, Δw2, Δw3)": [],
    "Pesos Después (w1, w2, w3)": []
}

convergencia_alcanzada = False

# Ciclo de entrenamiento para ajustar los pesos hasta converger
while not convergencia_alcanzada:
    ajuste_total = 0  # Variable para verificar la magnitud de ajuste total de pesos
    errores = []  # Almacenar los errores de cada ciclo
    for i in range(len(X)):
        # Peso antes
        peso_antes = pesos.copy()

        # Cálculo de salida
        salida = calcular_salida(X[i], peso_antes)

        # Salida Decimal (S.D.) - Convertir el patrón binario a decimal (esto será la salida deseada d(k))
        sd = binario_a_decimal(X[i])

        # Error (d - y)
        error = sd - salida

        # Ajuste de pesos
        delta_pesos = alpha * error * X[i]

        # Actualizar pesos
        pesos += delta_pesos

        # Suma de ajustes para evaluar convergencia
        ajuste_total += np.sum(np.abs(delta_pesos))

        # Guardamos los datos en la estructura
        data["Patrón Binario"].append(list(map(int, X[i])))  # Convertir de np.int64 a int
        data["S.D. (z(k))"].append(int(sd))  # Esto es d(k) calculado a partir del patrón binario
        data["Peso Antes (w1, w2, w3)"].append(np.round(peso_antes, 4).tolist())  # Convertir a lista legible
        data["Salida (S(k))"].append(np.round(salida, 4))
        data["Error (d - y)"].append(np.round(error, 4))
        data["Ajuste de Pesos (Δw1, Δw2, Δw3)"].append(np.round(delta_pesos, 4).tolist())  # Convertir a lista legible
        data["Pesos Después (w1, w2, w3)"].append(np.round(pesos, 4).tolist())  # Convertir a lista legible

        # Almacenar errores para verificar convergencia
        errores.append(abs(error))

    # Verificamos si el ajuste es menor que la tolerancia para detener, y si los errores son menores a la tolerancia
    if ajuste_total < tolerancia and all(e < tolerancia for e in errores):
        convergencia_alcanzada = True

# Convertir a DataFrame para visualizar en consola
df = pd.DataFrame(data)

# Mejorar la presentación con separación en tablas de 10 entradas
def print_table(df, batch_size=10):
    for start in range(0, len(df), batch_size):
        end = start + batch_size
        batch_df = df[start:end]
        # Encabezado general
        print("=" * 145)
        print(f"{'Patrón Binario':^20}{'S.D. (z(k))':^10}{'Peso Antes':^30}{'Salida (S(k))':^15}{'Error (d - y)':^12}{'Ajuste de Pesos':^30}{'Pesos Después':^30}")
        print("=" * 145)
        for index, row in batch_df.iterrows():
            # Imprimir cada fila
            print(f"{str(row['Patrón Binario']):^20}{str(row['S.D. (z(k))']):^10}{str(row['Peso Antes (w1, w2, w3)']):^30}{str(row['Salida (S(k))']):^15}{str(row['Error (d - y)']):^12}{str(row['Ajuste de Pesos (Δw1, Δw2, Δw3)']):^30}{str(row['Pesos Después (w1, w2, w3)']):^30}")
        print("=" * 145)

# Imprimir la tabla en partes de 10 entradas
print_table(df, batch_size=10)
