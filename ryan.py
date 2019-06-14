import gzip
import matplotlib.pyplot as plt
#matplotlib inline
import numpy as np
import numpy.ma as ma
from numpy import matmul as mul
import scipy as sp
from sklearn import manifold
from sklearn import neighbors
from sklearn import datasets
import struct


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


# Compute the complete nearest neighbors array
#
# Inputs: X      --> training data
#         n_nbrs --> the number of nearest neighbors to search for
#         n_comp --> the number of components
def compute_nearest_neighbors(X, n_nbrs, n_comp):
    print("Computing nearest neighbors...")
    tree = neighbors.BallTree(X, leaf_size=n_comp)
    dist, ind = tree.query(X, k=n_nbrs)
    print("Complete nearest neighbor index: ", np.shape(ind))
    return dist, ind


#------------------------------------------------------------------------------
# Inputs: x --> the basis vector (784 x 1)
#         E --> eta is the matrix of nearest neighbors (k x 784)
# 
# Outputs: C --> the covariance matrix of the nearest neighbors of the vector x 
#
# Iterate theough the nearest neighbors and compute the  
#
def calc_covariance(Eta):
    print("Calculate covariance:")
    #print("Eta: ", np.shape(Eta))
    
    # Compute the local covariance matrix
    #C = np.cov(Eta)
    C = mul(Eta.T, Eta)
    #print("Local covariance matrix C: ", np.shape(C))

    # Regularize the covariance matrix because otherwise it is singular
    I = np.identity(np.shape(C)[0])
    eps = .001 * np.trace(C)
    C = C + eps * I
    
    return C


#------------------------------------------------------------------------------
# Clear non-neighbor weight entries from the weight vector
#
# Inputs:  nbrs --> the list of nearest neighbors
#          wght --> the calculated weight vector
# 
# Outputs: wght --> the adjusted weight vector    
#
def clear_non_neighbors(nbrs, wght):
    print("Clear non-nieghbors:")
    print("wght: ", np.shape(wght))
    
    # Create mask vector
    dim = np.shape(wght)[0]
    mask = np.ones(dim)                        # ones mean, discard the value
    print("Mask: ", np.shape(mask))

    # Apply zeros to the nearest neighbors so sum of weights = 1
    dim_nbrs = np.shape(nbrs)[0]
    print("dim nbrs: ", dim_nbrs)
    for i in range(0, dim_nbrs):
        mask[nbrs[i]] = 0;
        #print("nbr: ", nbrs[0,i])
        
    # mask off the non-neighbor weights
    w = ma.masked_array(wght, mask)
    print(w)
    
    return w
    

#------------------------------------------------------------------------------
# Scale weight values for nearest neighbors
#
# Inputs:  wght --> the wight vector with non-neighbors blanked out
# 
# Outputs: w    --> the adjusted weight vector    
#
def scale_neighbors(wght):
    sumw = np.sum(wght)
    w = wght / sumw
    print("Sum of weights: ", np.sum(w))
    return w


#------------------------------------------------------------------------------
# Build the matrix of nearest neighbors
# 
# Inputs: X    --> the source data matrix X to get the neighbors from
#         nbrs --> the list of neighbor indices
#
# Outputs: Eta --> the nearest neighbor matrix for Xi
#
# Use the neighbors list to get the corresponding vector from X and put the 
# vectors together in a nearest neighbor matrix
#
def build_nbrs_matrix(X, nbrs, k):
    print("Build nearest neighbors matrix...")
    # Build a matrix of the nearest neighbors
    nbrs = np.delete(nbrs, 0, axis=0)   # remove Xi
    print("nbrs: ", np.shape(nbrs))
    
    D = np.shape(X)[1]                # set number of rows for Eta
    N = np.shape(nbrs)[1]         # set columns for Eta. Subtract 1 because index from ball tree includes the vector itself
    Eta = np.zeros([D, k])
    
    #print("D: ", D, " N: ", N)
    #print("Eta: ", np.shape(Eta))
    
    # Get each row vector from X corresponding to the nearest neighbor
    for n in range(1, N):
        Eta[:,n-1] = X[nbrs[0][n]]
        #print("nbr ", nbrs[0][n])
        
        
    #print("Eta: ", np.shape(Eta))    
    #print(Eta)
    return Eta


#------------------------------------------------------------------------------
# Center the matrix
#
#
#
#
def center_nbrs_matrix(x, Eta):
    print("Center the neighbors matrix...")
    n = np.shape(Eta)[1]
    d = np.shape(x)[0]
    #print("n: ", n)
    #print("d: ", d)
    #print("x: ", np.shape(x)) 
    sub = np.tile(x, (n,1))
    #print("sub: ", np.shape(sub))
    #print("Eta: ", np.shape(Eta))
    eta = Eta - sub.T
    return eta


#------------------------------------------------------------------------------
# Determine the reconstruction weights for each X
#
# Inputs:  X    --> Data Matrix
#          i    --> Index for vector xi we want to construct weights for
#          tree --> nearest neighbors tree
#          k    --> number of nearest neighbors  
# 
# Outputs: W    --> the weight matrix
#
def construct_weight_vector(X, i, nbrs, k):
    # Setup matrices and variables
    print("Construct weight matrix")
    #print("k = ", k)
    #print("Image #", i)
    #print("Neighbors = ", nbrs)

    # Build the matrix of nearest neighbors
    Eta = build_nbrs_matrix(X, nbrs, k)
    
    # Center the data
    Eta = center_nbrs_matrix(X[i,:], Eta)
    
    # Compute local covariance matrix for the vector v in X and all neighbors
    C = calc_covariance(Eta)

    # Solve the linear system with constraint that rows of weights sum to one
    b = np.ones(np.shape(C)[1])  # build the constant vector
    w = np.linalg.solve(C, b)    # compute solution
    
    # Apply first constraint to zero out non-neighbor weights
    #w = clear_non_neighbors(nbrs, w)
    
    # Apply second constraint to scale valid neighbors
    w = scale_neighbors(w)
    
    #print("Weight vector constructed...  w: ", np.shape(w))
    print("w = ", w)
    
    return w

#------------------------------------------------------------------------------
# Calculate the sum of elements in a weight row for matrix M
#
# Inputs:   W
#             
#
#
#
#
def sum_weight_row(W, i, j):
    s = 0
    for k in range(0, np.shape(W)[0]):
        s = s + W[k,i] * W[k,j] 
    return s


#------------------------------------------------------------------------------
#
#
#
#
#
def compute_embedding_components(W, d):
    print("Compute embedding components:")
    #create sparse matrix M = (I-W)'*(I-W)
    
    # Create matrix M
    print("Create matrix M...")
    M = np.zeros(np.shape(W))
    d = np.shape(M)[0]
    n = np.shape(M)[1]
    
    print("d: ", d, " n: ", n)
    
    for i in range(0, d):
        for j in range(0, n):
            delta = 1 if i==j else 0
            M[i,j] = delta - W[i,j] - W[i,j] + sum_weight_row(W, i, j)
        print("M[",i,",",j,"] = ", M[i,j])        
    

    # 
    U, S, Vt = np.linalg.svd(M)
    
    #print("U: ", np.shape(U))
    #print(U,"\n")
    #print("S: ", np.shape(S))
    #print("Vt: ", np.shape(Vt))
    
    d = np.shape(U)[1]
    E = U[:,d-3:d-1]
    print("E: ", np.shape(E))
    print(E)
    return Y


#------------------------------------------------------------------------------
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
    
    # Testing
    #tree = neighbors.BallTree(X, leaf_size=n_components)
    #dist, ind = tree.query(X[:1], k=n_neighbors)
    #print("ind: ", ind)
    
    # Primetime
    #n_neighbors = n_neighbors + 1
    dist, ind = compute_nearest_neighbors(X, n_neighbors + 1, n_components)
    
    # Step 2: Construct the weight matrix
    D = np.shape(X)[0]
    N = np.shape(X)[1]
    W = np.zeros((n_neighbors, D))  # subtract 1 from D because nearest neighbor returns the vector itself
    
    print("X: ", np.shape(X))
    print("N: ", N, " D: ", D)
    print("W: ", np.shape(W))

    # Loop through each image and construct its weight matrix 
    for i in range(0, D):
        print("Image: ",i)
        w = construct_weight_vector(X, i, ind, n_neighbors)
        W[:,i] = w[:]
    
    # Step 3: Compute vectors that are reconstructed by weights
    Y = compute_embedding_components(W, 2)
    
    # Calculate the error
    
    err = 0.001    # placeholder
    return Y, err




# MAIN
# -------------------------------------------------------------------------------------------
np.set_printoptions(precision=2, edgeitems=10)

raw_train = read_idx("train-images-idx3-ubyte.gz")
train_data = np.reshape(raw_train, (60000, 28*28))
train_label = read_idx("train-labels-idx1-ubyte.gz")

# Train algorithm and calculate landmark graph
X = train_data[train_label == 8]

# Test by using scikit class
#Y, err = manifold.locally_linear_embedding(X, n_neighbors=10, n_components=2)

# Verify by using our custom class 
Y, err = locally_linear_embedding(X, n_neighbors=10, n_components=2)

print("Y: ", np.shape(Y))

# Find landmarks
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












