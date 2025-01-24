from typing import NamedTuple

import numpy
import numpy as np


type Number = numpy.ScalarType
type SquareMatrix[N] = np.matrix[(N, N), Number]


class LUDecomposition[N](NamedTuple):
    lower: SquareMatrix[N]
    upper: SquareMatrix[N]


def lu_decompose[N](matrix: SquareMatrix[N], lower: SquareMatrix[N] = None) -> LUDecomposition[N]:
    assert len(matrix.shape) == 2
    assert matrix.shape[0] == matrix.shape[1]
    n: int = matrix.shape[0]
    assert n > 0

    if lower is None:
        matrix = matrix.copy()
        lower = np.asmatrix(np.identity(n, matrix.dtype))

    # divide the first row to set its first entry to '1'
    divider = matrix[0, 0]
    if divider != 1:
        matrix[0] /= divider
        lower[0, 0] = divider

    # base-case: if matrix is 1-by-1, done
    if n == 1:
        return LUDecomposition(
            lower,
            matrix,
        )

    # linear combine the remaining rows to set their first entries to '0'
    for row_index in range(1, n):
        multiplier = matrix[row_index, 0]
        if multiplier == 0:
            continue

        matrix[row_index] -= multiplier * matrix[0]
        lower[row_index, 0] = multiplier

    # decompose the remaining sub-matrix in-place
    lu_decompose(matrix[1:, 1:], lower[1:, 1:])

    return LUDecomposition(
        lower,
        matrix,
    )


if __name__ == "__main__":
    my_matrix = numpy.asmatrix([
        [3, -6, -3],
        [2, 0, 6],
        [-4, 7, 4]
    ], dtype=float)
    decomposition = lu_decompose(my_matrix)

    print(decomposition.lower)
    print(decomposition.upper)
    print()
    round_trip = decomposition.lower @ decomposition.upper
    print(round_trip)
    print((round_trip == my_matrix).all())
