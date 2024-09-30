import numpy as np  # Importamos numpy para manejar cálculos matemáticos y operaciones con arrays
import pandas as pd  # Importamos pandas para estructurar y visualizar los datos en forma de tablas

# Parámetros
alpha = 0.3  # Tasa de aprendizaje, usada para ajustar los pesos en cada iteración
tolerancia = 1e-5  # Umbral de tolerancia, para verificar cuando el ajuste en los pesos es suficientemente pequeño y detener el entrenamiento
# Matriz de entrada (10 patrones binarios con 4 columnas, representando el conjunto de datos de entrada)
X = np.array([[0, 0, 0, 1],  # Cada fila representa un patrón binario, en este caso, con 4 características
              [0, 1, 0, 0],
              [0, 1, 1, 1],
              [1, 0, 0, 0],
              [1, 0, 1, 1],
              [1, 1, 0, 0],
              [1, 1, 1, 1],
              [0, 0, 0, 0],
              [1, 0, 0, 1],
              [1, 0, 1, 0]])


# Función para predecir la salida, multiplica el patrón binario con los pesos actuales
def calcular_salida(X, pesos):
    return np.dot(X, pesos)  # Producto punto entre el vector de entrada y el vector de pesos


# Función que convierte un patrón binario a decimal (el valor decimal será la salida deseada)
def binario_a_decimal(binario):
    # Convertimos el patrón binario en decimal aplicando las potencias de 2, por ejemplo [1, 0, 1, 1] -> 11 en decimal
    return binario[0] * 2 ** 3 + binario[1] * 2 ** 2 + binario[2] * 2 ** 1 + binario[3] * 2 ** 0


# Inicialización de los pesos, estos serán ajustados durante el proceso de aprendizaje
# Comenzamos con un conjunto de valores aleatorios o predefinidos
pesos = np.array([0.631, 0.840, 0.394, 0.576])

# Estructura de datos donde guardaremos los resultados de cada iteración para mostrarlo como tabla
data = {
    "Patrón Binario": [],  # Columna para almacenar los patrones de entrada (X)
    "S.D. (z(k))": [],  # Columna para almacenar la salida decimal (convertida del patrón binario)
    "Peso Antes (w1, w2, w3, w4)": [],  # Pesos antes de aplicar el ajuste
    "Salida (S(k))": [],  # Salida calculada S(k) antes del ajuste
    "Error (d - y)": [],  # Diferencia entre la salida deseada (d) y la salida calculada (y)
    "Ajuste de Pesos (Δw1, Δw2, Δw3, Δw4)": [],  # Ajuste calculado para los pesos en esa iteración
    "Pesos Después (w1, w2, w3, w4)": []  # Pesos después del ajuste
}

convergencia_alcanzada = False  # Bandera que indica si hemos alcanzado la convergencia (ajuste total cercano a 0)

# Bucle principal de aprendizaje, el proceso continuará hasta que el ajuste total sea menor que la tolerancia
while not convergencia_alcanzada:
    ajuste_total = 0  # Variable para almacenar la magnitud total del ajuste de pesos en esta iteración
    errores = []  # Almacenamos los errores en cada ciclo de entrenamiento

    for i in range(len(X)):  # Recorremos cada patrón de entrada (X)
        # Guardamos los pesos antes de cualquier ajuste
        peso_antes = pesos.copy()

        # Calculamos la salida S(k), producto punto entre el patrón binario y los pesos
        salida = calcular_salida(X[i], peso_antes)

        # Convertimos el patrón binario a decimal para usarlo como la salida deseada d(k)
        sd = binario_a_decimal(X[i])

        # Calculamos el error, que es la diferencia entre la salida deseada y la salida calculada
        error = sd - salida

        # Calculamos el ajuste de pesos basado en el error y el patrón de entrada (Regla delta: α(d - y)X)
        delta_pesos = alpha * error * X[i]

        # Actualizamos los pesos sumando el ajuste calculado
        pesos += delta_pesos

        # Acumulamos el ajuste total de los pesos para verificar convergencia
        ajuste_total += np.sum(np.abs(delta_pesos))

        # Guardamos los resultados en las respectivas columnas de la tabla
        data["Patrón Binario"].append(list(map(int, X[i])))  # Guardamos el patrón binario como enteros
        data["S.D. (z(k))"].append(int(sd))  # Guardamos la salida decimal (convertida del binario)
        data["Peso Antes (w1, w2, w3, w4)"].append(
            np.round(peso_antes, 4).tolist())  # Guardamos los pesos antes del ajuste
        data["Salida (S(k))"].append(np.round(salida, 4))  # Guardamos la salida calculada
        data["Error (d - y)"].append(np.round(error, 4))  # Guardamos el error
        data["Ajuste de Pesos (Δw1, Δw2, Δw3, Δw4)"].append(
            np.round(delta_pesos, 4).tolist())  # Guardamos el ajuste de los pesos
        data["Pesos Después (w1, w2, w3, w4)"].append(
            np.round(pesos, 4).tolist())  # Guardamos los pesos después del ajuste

        # Guardamos los errores para verificar si todos son suficientemente pequeños y detener el entrenamiento
        errores.append(abs(error))

    # Si el ajuste total de todos los pesos es menor que la tolerancia y los errores son lo suficientemente pequeños, detenemos
    if ajuste_total < tolerancia and all(e < tolerancia for e in errores):
        convergencia_alcanzada = True  # Convergencia alcanzada

# Convertimos los datos almacenados a un DataFrame para facilitar la visualización en consola
df = pd.DataFrame(data)


# Función para imprimir la tabla en partes de 10 filas, con títulos claros para cada columna
def print_table(df, batch_size=10):
    for start in range(0, len(df), batch_size):  # Iteramos en bloques de 10 filas
        end = start + batch_size  # Definimos el rango para cada bloque de impresión
        batch_df = df[start:end]  # Extraemos el bloque de filas
        # Imprimimos los encabezados y subtítulos
        print("=" * 170)
        print(
            f"{'Patrón Binario':^30}{'S.D. (z(k))':^12}{'Pesos Iniciales':^40}{'Salida (S(k))':^15}{'Error (d - y)':^15}{'Ajuste de Pesos':^40}{'Pesos Finales':^40}")
        print(
            f"{'':^30}{'':^12}{'(w1, w2, w3, w4)':^40}{'':^15}{'':^15}{'(Δw1, Δw2, Δw3, Δw4)':^40}{'(w1, w2, w3, w4)':^40}")
        print("=" * 170)
        # Imprimimos cada fila de datos
        for index, row in batch_df.iterrows():
            print(
                f"{str(row['Patrón Binario']):^30}{str(row['S.D. (z(k))']):^12}{str(row['Peso Antes (w1, w2, w3, w4)']):^40}{str(row['Salida (S(k))']):^15}{str(row['Error (d - y)']):^15}{str(row['Ajuste de Pesos (Δw1, Δw2, Δw3, Δw4)']):^40}{str(row['Pesos Después (w1, w2, w3, w4)']):^40}")
        print("=" * 170)


# Imprimimos la tabla en bloques de 10 filas
print_table(df, batch_size=10)
