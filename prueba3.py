import numpy as np
import matplotlib.pyplot as plt  # Importar la librería para gráficos


# Definir función de activación (lineal en este caso)
def activation_function(x):
    return x


# Función para convertir patrones binarios a decimales
def binary_to_decimal(X):
    """
    Convierte una lista de patrones binarios en sus valores decimales correspondientes.
    - X: Matriz de patrones binarios (cada fila es un patrón binario).
    """
    return np.array([int("".join(str(bit) for bit in pattern), 2) for pattern in X])


# Clase que implementa el modelo ADELINE
class Adeline:
    def __init__(self, input_size, learning_rate=0.3):
        """
        Constructor que inicializa el modelo ADELINE.
        - input_size: número de entradas (en este caso 3, porque hay 3 características binarias).
        - learning_rate: tasa de aprendizaje, controla cuánto cambian los pesos en cada iteración.
        """
        # Inicializamos los pesos aleatoriamente para 10 patrones binarios y 3 entradas (input_size).
        self.weights_list = np.random.rand(10,
                                           input_size)  # Se generan 10 conjuntos de pesos diferentes para cada patrón binario.

        # Asignamos la tasa de aprendizaje.
        self.learning_rate = learning_rate  # La tasa de aprendizaje se usará en la regla delta para ajustar los pesos.

    # Método para predecir la salida del modelo, dadas las entradas (X) y los pesos (weights).
    def predict(self, X, weights):
        """
        Este método calcula el producto punto entre las entradas X y los pesos.
        - X: patrón binario (entrada).
        - weights: pesos asociados a ese patrón.
        """
        return activation_function(np.dot(X, weights))  # np.dot realiza el producto punto entre X y los pesos.

    # Método para entrenar el modelo
    def train(self, X, y, epochs=1):
        """
        Entrena el modelo ADELINE ajustando los pesos para minimizar el error.
        - X: entradas (patrones binarios).
        - y: salidas deseadas (valores decimales).
        - epochs: número de veces que se repite el proceso de entrenamiento (iteraciones). Aquí se hace una sola iteración (epochs=1).
        """

        # Almacenamos las salidas predichas para graficar
        outputs = []

        # Imprimimos la cabecera de la tabla que mostrará el patrón binario y su salida decimal correspondiente.
        print(f"{'Iteración':^10} | {'Patrón Binario':^18} | {'Salida Decimal':^15}")
        print("-" * 50)

        # Bucle para imprimir cada patrón binario con su salida correspondiente.
        for i in range(len(X)):
            print(f"{i + 1:^10} | {str(X[i]):^18} | {y[i]:^15}")

        # Imprimimos la cabecera para mostrar los pesos antes del ajuste.
        print("\nPesos Antes del Ajuste (diferentes para cada iteración)")
        print(f"{'W1':^10} | {'W2':^10} | {'W3':^10}")
        print("-" * 30)

        # Bucle para mostrar los pesos iniciales antes de cada ajuste, para cada patrón.
        for i in range(len(X)):  # Itera sobre los 10 patrones binarios.
            print(
                f"{self.weights_list[i][0]:^10.4f} | {self.weights_list[i][1]:^10.4f} | {self.weights_list[i][2]:^10.4f}")

        # Imprimimos la cabecera para mostrar las salidas calculadas y los errores.
        print("\nSalida Calculada y Error")
        print(f"{'Patrón Binario':^18} | {'Salida (X.W)':^15} | {'Error (e)':^10}")
        print("-" * 60)

        # Bucle para calcular y mostrar las salidas y los errores para cada patrón.
        for i in range(len(X)):
            # Calculamos la salida predicha con la función predict (usando las entradas y los pesos correspondientes).
            output = self.predict(X[i], self.weights_list[i])
            outputs.append(output)  # Guardamos la salida para el gráfico
            # Calculamos el error como la diferencia entre la salida deseada y la salida predicha.
            error = y[i] - output
            # Mostramos el patrón binario, la salida predicha y el error calculado.
            print(f"{str(X[i]):^18} | {output:^15.4f} | {error:^10.4f}")

        # Imprimimos la cabecera para mostrar los ajustes de los pesos usando la Regla Delta.
        print("\nAjuste de Pesos (Regla Delta)")
        print(f"{'ΔW1':^10} | {'ΔW2':^10} | {'ΔW3':^10}")
        print("-" * 30)

        # Bucle para calcular y ajustar los pesos para cada patrón.
        for i in range(len(X)):
            # Calculamos nuevamente la salida predicha.
            output = self.predict(X[i], self.weights_list[i])
            # Calculamos el error como antes.
            error = y[i] - output
            # Calculamos el ajuste en los pesos usando la fórmula de la regla delta:
            delta_w = self.learning_rate * error * X[i]
            # Mostramos los ajustes de pesos para cada peso (W1, W2, W3).
            print(f"{delta_w[0]:^10.4f} | {delta_w[1]:^10.4f} | {delta_w[2]:^10.4f}")
            # Actualizamos los pesos sumando el ajuste calculado (Δw_j).
            self.weights_list[i] += delta_w

        # Imprimimos la cabecera para mostrar los pesos después del ajuste.
        print("\nPesos Después del Ajuste")
        print(f"{'W1':^10} | {'W2':^10} | {'W3':^10}")
        print("-" * 30)

        # Bucle para mostrar los pesos actualizados después del ajuste.
        for i in range(len(X)):
            print(
                f"{self.weights_list[i][0]:^10.4f} | {self.weights_list[i][1]:^10.4f} | {self.weights_list[i][2]:^10.4f}")

        # Después del entrenamiento, graficamos los resultados
        self.plot_graph(X, outputs, y)

    # Método para graficar
    def plot_graph(self, X, outputs, y):
        """
        Crea un gráfico que muestra los puntos de las salidas deseadas y las salidas predichas,
        junto con la línea que representa la fórmula de ajuste.
        """
        # Configuración de los puntos y la línea
        plt.figure(figsize=(6, 5))

        # Graficamos las salidas predichas como puntos azules
        plt.scatter(outputs, y, color='blue', label='Salidas predichas')

        # Dibujamos una línea que representa la fórmula de ajuste de la salida
        line_x = np.linspace(-1.5, 2, 100)  # Valores para graficar la línea
        line_y = line_x  # La línea es lineal para este caso
        plt.plot(line_x, line_y, 'b--', linewidth=3, label=r'$\bar{y} = w_1 \cdot x_1 + w_2 \cdot x_2 + w_3 \cdot x_3$')

        # Configuración adicional
        plt.axhline(0, color='black', linewidth=1)
        plt.axvline(0, color='black', linewidth=1)
        plt.title('Gráfico de la fórmula y las salidas predichas')
        plt.xlabel('Salidas predichas')
        plt.ylabel('Salidas deseadas')
        plt.legend()
        plt.grid(True)
        plt.show()


# Definimos los patrones binarios de entrada
X = np.array([[0, 1, 0],  # Primer patrón binario
              [0, 0, 1],  # Segundo patrón binario
              [0, 1, 0],  # Tercer patrón binario
              [0, 1, 1],  # Cuarto patrón binario
              [1, 0, 0],  # Quinto patrón binario
              [1, 0, 1],  # Sexto patrón binario
              [1, 1, 0],  # Séptimo patrón binario
              [1, 1, 1],  # Octavo patrón binario
              [1, 0, 0],  # Noveno patrón binario
              [1, 0, 1]])  # Décimo patrón binario

# Calculamos las salidas decimales a partir de los patrones binarios
y = binary_to_decimal(X)

# Creamos una instancia del modelo ADELINE con 3 entradas (input_size=3) y una tasa de aprendizaje de 0.3.
adeline_model = Adeline(input_size=3, learning_rate=0.3)

# Entrenamos el modelo utilizando los patrones de entrada (X) y las salidas deseadas (y).
adeline_model.train(X, y, epochs=1)  # Solo hacemos una iteración (epochs=1).
