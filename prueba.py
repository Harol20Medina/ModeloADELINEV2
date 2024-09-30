import numpy as np
import pandas as pd

# Parámetros
alpha = 0.3
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
    return binario[0] * 2 ** 2 + binario[1] * 2 ** 1 + binario[2] * 2 ** 0


# Inicializar pesos
pesos = np.array([[0.631, 0.840, 0.394],
                  [0.631, 0.840, 0.576],
                  [0.631, 1.188, 0.576],
                  [0.631, 1.559, 0.947],
                  [1.642, 1.559, 0.947],
                  [2.365, 1.559, 1.670],
                  [3.073, 1.643, 1.421],
                  [2.365, 1.559, 1.469],
                  [2.856, 1.643, 1.421],
                  [3.073, 1.643, 1.638]])

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

# Proceso de aprendizaje
for i in range(len(X)):
    # Peso antes
    peso_antes = pesos[i]

    # Cálculo de salida
    salida = calcular_salida(X[i], peso_antes)

    # Salida Decimal (S.D.) - Convertir el patrón binario a decimal (esto será la salida deseada d(k))
    sd = binario_a_decimal(X[i])

    # Error (d - y)
    error = sd - salida

    # Ajuste de pesos
    delta_pesos = alpha * error * X[i]

    # Actualizar pesos
    peso_despues = peso_antes + delta_pesos

    # Guardamos los datos en la estructura
    data["Patrón Binario"].append(list(map(int, X[i])))  # Convertir de np.int64 a int
    data["S.D. (z(k))"].append(int(sd))  # Esto es d(k) calculado a partir del patrón binario
    data["Peso Antes (w1, w2, w3)"].append(np.round(peso_antes, 4).tolist())  # Convertir a lista legible
    data["Salida (S(k))"].append(np.round(salida, 4))
    data["Error (d - y)"].append(np.round(error, 4))
    data["Ajuste de Pesos (Δw1, Δw2, Δw3)"].append(np.round(delta_pesos, 4).tolist())  # Convertir a lista legible
    data["Pesos Después (w1, w2, w3)"].append(np.round(peso_despues, 4).tolist())  # Convertir a lista legible

    # Actualizamos los pesos
    pesos[i] = peso_despues

# Convertir a DataFrame para visualizar en consola
df = pd.DataFrame(data)


# Mejorar la presentación con separación en líneas
def print_table(df):
    # Encabezado general
    print("=" * 145)
    print(
        f"{'Patrón Binario':^20}{'S.D. (z(k))':^10}{'Peso Antes':^30}{'Salida (S(k))':^15}{'Error (d - y)':^12}{'Ajuste de Pesos':^30}{'Pesos Después':^30}")
    print("=" * 145)

    for index, row in df.iterrows():
        # Imprimir cada fila
        print(
            f"{str(row['Patrón Binario']):^20}{str(row['S.D. (z(k))']):^10}{str(row['Peso Antes (w1, w2, w3)']):^30}{str(row['Salida (S(k))']):^15}{str(row['Error (d - y)']):^12}{str(row['Ajuste de Pesos (Δw1, Δw2, Δw3)']):^30}{str(row['Pesos Después (w1, w2, w3)']):^30}")
        print("-" * 145)


# Imprimir la tabla
print_table(df)
