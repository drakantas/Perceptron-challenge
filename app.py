from typing import Callable, List
from random import uniform


# Implementación de hardlim
# si n >= 0: retornar 1; sino: retornar 0
hardlims = lambda n: 1 if n >= 0 else -1


class Perceptron:
    def __init__(self, filter_func: Callable, weight_array: List):
        # Cantidad de iteraciones realizadas (épocas)
        self.iterations = 0

        # Función filtro
        self.filter_func = filter_func

        # Bias (Umbral)
        self.bias = 0

        # Vector con los pesos
        self.weight_array = weight_array

    def run(self, data: List, iterations: int = 1):
        """
        Formato de data brindada:
        [
            [
                [
                    forma,
                    textura,
                    peso
                ], valor_esperado
            ],
            ...
        ]
        """
        while True:
            # Aumentar contador de épocas ó iteraciones en 1
            self.iterations += 1

            for e in data:
                self.train(e[0], e[1])

            # Si se ha alcanzado la cantidad de épocas brindadas, salir del loop
            if self.iterations == iterations:
                break

    def train(self, input_: List, expected_value: int):
        output = self.filter_func(input_, self.weight_array, self.bias)

        error = expected_value - output

        # Aprender de data brindada
        self._learn(input_, error)

    def _learn(self, data: List, error: int):
        # El threshold ó umbral deberá de ser actualizado
        self.bias += error

        # También el vector de pesos
        self.weight_array = [data[i] + error * self.weight_array[i] for i, _ in enumerate(self.weight_array)]


def fruit_pattern_classifier(weird_orange: bool = False):
    def filter_func(input_data: List, weight_array: List, bias: int) -> int:
        if len(input_data) != len(weight_array):
            raise ValueError

        # Ecuación matemática del perceptrón
        return hardlims(bias + sum([input_data[i] * weight_array[i] for i, _ in enumerate(input_data)]))

    perceptron = Perceptron(filter_func, [0, 0, 0])

    data = [
        [
            [
                1,
                1,
                -1
            ], 1  # Manzana
        ],
        [
            [
                1,
                -1,
                -1
            ], -1  # Naranja
        ]
    ]

    if weird_orange:
        data.append([
            [
                -1,
                -1,
                -1
            ], 1  # Una naranja elíptica
        ])

    perceptron.run(data)

    print('''
        Perceptron Challenge.
        -----
        Umbral: {0}
        Vector de pesos: [{1}]
    '''.format(perceptron.bias, ';'.join(list(map(lambda e: str(e), perceptron.weight_array)))))


if __name__ == '__main__':
    fruit_pattern_classifier(weird_orange=False)
