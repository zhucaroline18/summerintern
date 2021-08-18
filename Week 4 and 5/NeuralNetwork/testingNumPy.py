import numpy as np 

m1 = np.array([[1,2,3],[4,5,6]])
v1 = np.array([1,2,3])
r1 = m1.dot(v1)
print(r1)

v2 = np.array([10,12])
print(r1+v2)

print()
print (v1)
v1.reshape(-1,1)
print(v1)
print()

r2 = m1.dot(v1.reshape(-1,1))
print(r2)

print (r2+v2)

print()
l1 = [1,2,3]
print(l1)
l2 = np.array([1,2,3])
print(l2)
l3 = np.array(l2).reshape(-1,1)
print(l3)

# convert a vector to a matrix for operations to avoid unexpected result

