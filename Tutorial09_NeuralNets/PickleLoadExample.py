import pickle
with open('mnist.dat','rb') as f:
   mnist = pickle.load(f)
print(mnist.data)
print(mnist.target)
