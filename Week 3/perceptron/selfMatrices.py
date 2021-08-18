#creating and implementing own matrix class with same functions to see how it works 

class Matrix:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.data = [[0.0 for j in range(cols)] for i in range(rows)]

    @staticmethod #do this in python for static methods 
    def add(matrixA, matrixB):
        if (matrixA.rows!=matrixB.rows):
            raise Exception("Matrix rows not match")
        if (matrixA.cols!=matrixB.cols):
            raise Exception("Matrix rows not match")

        ret = Matrix(matrixA.rows, matrixA.cols)
        for i in range (matrixA.rows):
            for j in range(matrixA.cols):
                ret.data[i][j]=matrixA.data[i][j]+matrixB.data[i][j]
        return ret

    def printMatrix(self):
        print("[")
        for i in range(self.rows):
            print(self.data[i], ", ")
        print("]")

    @staticmethod
    def sub(matrixA, matrixB):
        if (matrixA.rows!=matrixB.rows):
            raise Exception("Matrix rows not match")
        if (matrixA.cols!=matrixB.cols):
            raise Exception("Matrix rows not match")

        ret = Matrix(matrixA.rows, matrixA.cols)
        for i in range (matrixA.rows):
            for j in range(matrixA.cols):
                ret.data[i][j]=matrixA.data[i][j]-matrixB.data[i][j]
        return ret
    
    @staticmethod
    def dot(matrixA, matrixB):
        if (matrixA.cols!=matrixB.rows):
            raise Exception("Matrix can't be multiplied")
        
        ret = Matrix(matrixA.rows, matrixB.cols)

        for i in range(ret.rows):
            for j in range(ret.cols):
                ##now we calculate what that number will be
                result = 0
                for x in range(matrixA.cols):
                    result += matrixA.data[i][x]*matrixB.data[x][j]

                    ##for y in range(matrixB.rows):
                      ##  result += matrixA.data[i,x]*matrixB.data[y,j]
                ret.data[i][j]=result
        return ret