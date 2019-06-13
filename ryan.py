import gzip
import matplotlib.pyplot as plt
#matplotlib inline
import numpy as np
import scipy as sp
from sklearn import manifold
from sklearn import neighbors
from sklearn import datasets
import struct
import sys

# Parse data from zipped mnist files into numpy arrays
# Modified from source: https://gist.github.com/tylerneylon/ce60e8a06e7506ac45788443f7269e40
#
# Input:  filename --> Zipped file to parse
#
# Output: return   --> Numpy array of unint8 data points
#
def read_idx(filename):
    with gzip.open(filename) as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
    

# Construct an nxm matrix of evenly distributed samples from an input sample set
#
# Inputs: Y    --> Sample matrix
#         n, m --> Landmark matrix dimensions
#
# Output: idx  --> nxm Landmark matrix
#
def find_landmarks(Y, n, m):
    xr = np.linspace(np.min(Y[:,0]), np.max(Y[:,0]), n)
    yr = np.linspace(np.min(Y[:,1]), np.max(Y[:,1]), m)
    xg, yg = np.meshgrid(xr, yr)
    idx = [0]*(n*m)
    for i, x, y in zip(range(n*m), xg.flatten(), yg.flatten()):
        idx[i] = int(np.sum(np.abs(Y-np.array([x,y]))**2, axis=-1).argmin())
    return idx


# Inputs: x --> the basis vector (784 x 1)
#         E --> eta is the matrix of nearest neighbors (k x 784)
# 
# Outputs: C --> the covariance matrix of the nearest neighbors of the vector x 
#
# Iterate theough the nearest neighbors and compute the  
#
def calc_covariance(x, Eta):
    print("Calculate covariance:")
    print("x: ", np.shape(x))
    print("E: ", np.shape(Eta))

    K = np.shape(Eta)[0]      # set K to iterate through each of the nearest neighbors
    C = np.zeros([K,K])     # set C as the KxK covariance matrix
    print("K: ", K)
    
    #Et = Eta.T
    #print("E[1]: \n", E[1], "\n")
    # center the data
    for k in range(0, K-1):
        Eta[k] = x - Eta[k]
    
    np.set_printoptions(threshold=sys.maxsize)
    print(Eta)
    print("Eta")
    
    print(Eta.T)
    print("Eta.T")
    
    # Compute the local covariance matrix
    C = np.cov(Eta)

    print("Local covariance matrix C: ", np.shape(C))
    print(C, "\n")

    return C


# Determine the reconstruction weights for each X
#
# Inputs:  X    --> Data Matrix
#          v    --> Index into the data matrix for the vector we want to construct weights for
#          tree --> nearest neighbors tree
#          k    --> number of nearest neighbors  
# 
# Outputs: W    --> the weight matrix
#
# Reconstruction errors measured by the cost function of the distances of every
# Xj for an Xi with weights applied to Xj
# Minimize the const function subject to two constraints:
# 1. Each data point Xi is reconstructed from its nearest neighbors only 
#    --> Wij = 0 if X Xj not in the set of nearest neighbors
# 2. Rows of the weight matrix sum to one
#
def construct_weight_matrix(X, v, neighbors, k):
    # Setup matrices and variables
    D = np.shape(X)[1]                              # get the dimension of X
    print("Construct weight matrix")
    print("X = ", np.shape(X))
    print("D = ", D)
    print("k = ", k)
    print("Image #", v, "(v)")
    print("Neighbors = ", neighbors)

    # Build a matrix of the K nearest neighbors
    n = neighbors[:,0]      # Set eta to the first nearest neighbor vector
    Eta = X[n]
    print("Eta: ", np.shape(Eta))
    
    # iterate through the rest of neighbors and concatenate them into Eta
    for i in range(1,k):      
        n = neighbors[:,i]
        Eta = np.row_stack((Eta, X[n]))
        print(np.shape(Eta))
    
    # Compute local covariance matrix for the vector v in X
    # and the Eta nearest neighbors
    C = calc_covariance(X[v,:], Eta)

    # Solve the linear system with constraint that rows of weights sum to one
    b = np.ones(np.shape(C)[1])  # build the constant vector
    W = np.linalg.solve(C, b)    # compute solution
    
    print("W: ", np.shape(W))
    print(W)
    
    # Apply second constraint if j is not a neighbor, set Wij = 0
    #for i in range(0, np.shape(X)[0]):
        

    return W




# TODO - We only have to implement this function, swap it with the manifold one below
# Seek a low-rank projection on an input matrix
#
# Inputs: X            --> Input matrix to reduce
#         n_neighbors  --> Maximum number of neighbors used for reconstruction
#         n_components --> Maximum number of linearly independent components for reconstruction
#
# Output: Y            --> Reconstructed vectors in a lower rank
#         err          --> (Optional implementation) Error margin of vectors
#
def locally_linear_embedding(X, n_neighbors, n_components):
    
    # Step 1: Find the nearest neighbors at for each sample
    tree = neighbors.BallTree(X, leaf_size=n_components)
    dist, ind = tree.query(X[:1], k=n_neighbors)
    # ind = indices of k=n_neighbors nearest neighbors
    # (nearest neighbors are ranked in order)
    # dist = distance k=n_neighbors nearest neighbors

    # Step 2: Construct a weight matrix that recreates X from its neighbors
    # Loop through each image and construct its weight matrix 
    W = np.zeros(np.shape(X))
    #for v in range(0, np.shape(X)[0] - 1)
    
    
    v = 1
    dist, ind = tree.query(X[:v], k=n_neighbors)
    W = construct_weight_matrix(X, v, ind, n_neighbors)


    
    # Step 3: Compute vectors that are reconstructed by weights
    
    
    Y = X[:,0:2]   # placeholder
    err = 0.001    # placeholder
    return Y, err




raw_train = read_idx("train-images-idx3-ubyte.gz")
train_data = np.reshape(raw_train, (60000, 28*28))
train_label = read_idx("train-labels-idx1-ubyte.gz")

# Train algorithm and calculate landmark graph
X = train_data[train_label == 8]
#Y, err = manifold.locally_linear_embedding(X, n_neighbors=10, n_components=2)
Y, err = locally_linear_embedding(X, n_neighbors=10, n_components=2)
landmarks = find_landmarks(Y, 5, 5)

# Plot the clustered data with landmarks overlaid
plt.scatter(Y[:,0], Y[:,1])
plt.scatter(Y[landmarks,0], Y[landmarks,1])

# Show the landmark samples in a 5x5 grid
fig = plt.figure(figsize=(15,15))
for i in range(len(landmarks)):
    ax = fig.add_subplot(5, 5, i+1)
    imgplot = ax.imshow(np.reshape(X[landmarks[i]], (28,28)), cmap=plt.cm.get_cmap("Greys"))
    imgplot.set_interpolation("nearest")
plt.show()












