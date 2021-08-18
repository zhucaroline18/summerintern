## trying out matrices with numpy to see how it differs 

import numpy as np

print (np.sum([1,2,3]))

matrixA = [[1,2,3],[4,5,6]]
vectorA = [1,2,3]
print(np.dot(matrixA,vectorA))
matrixB = [[1,2],[3,4],[5,6]]
print(np.dot(matrixA,matrixB))

matrixC = [[1,2,3],[1,2,3]]
print(np.subtract(matrixA,matrixC))

print(np.add(matrixA,matrixC))

print(np.add(matrixA,1))

print(matrixA)

print(np.array(matrixA))

