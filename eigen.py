import numpy as np
import numpy.linalg as linalg

A = np.random.random((3,3))
eigenValues,eigenVectors = linalg.eig(A)
print eigenValues
print eigenVectors

idx = eigenValues.argsort()[::-1]
eigenValues = eigenValues[idx]
eigenVectors = eigenVectors[:,idx]
print eigenValues
print eigenVectors
