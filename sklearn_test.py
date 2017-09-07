from numpy import array
from sklearn.linear_model.perceptron import Perceptron


# Definimos la data y su target para entrenar la red neuronal
data = [
    array([  # Vector de [forma;textura;peso]
        [1, 1, -1],  # Manzana
        [1, -1, -1],  # Naranja
        [1, 1, -1],  # Manzana
        [1, -1, -1],  # Naranja
    ]), array([
        1,
        -1,
        1,
        1
    ])
]

# Crear una instancia de Perceptron con un máximo de 100 épocas
perceptron = Perceptron(max_iter=1000)

# Entrenar la red neuronal
perceptron.fit(data[0], data[1])


def test_net_accuracy():
    # Probar la certeza de la red neuronal con la data previamente alimentada
    # para filtrar una manzana y una naranja.
    data = [
        array([
            [1, 1, -1],   # Manzana
            [1, -1, -1],  # Naranja
            [-1, -1, -1]  # Una naranja que es elíptica
        ]), array([
            1,
            1,
            1
        ])
    ]

    print('{0!s}% de precisión de la data vs los datos reales.'.format(perceptron.score(data[0], data[1]) * 100))

if __name__ == '__main__':
    test_net_accuracy()
