##testing my own matrix class I made 

import selfMatrices
import numpy as np

matrixA = selfMatrices.Matrix(2,3)
matrixB = selfMatrices.Matrix(3,2)

matrixA.data = [[1,2,3],[4,5,6]]
matrixB.data = [[2,3],[4, 5],[6,7]]

x = selfMatrices.Matrix.dot(matrixA, matrixB)

x.printMatrix()

npresult = np.dot(matrixA.data, matrixB.data)

print(npresult)