# Teaching Example: Collaborative Filtering with Matrix Factorization
# Andrew Chamberlain, Ph.D.
# October 2022
# Source: https://towardsdatascience.com/recommendation-system-matrix-factorization-d61978660b4b 

import numpy

# Define matrix factorization function with gradient descent. 
def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    '''
    R: Items rating matrix
    P: |U| * K (User features matrix)
    Q: |D| * K (Item features matrix)
    K: Number of latent features
    Steps: Number of iterations
    alpha: Learning rate
    beta: Regularization parameter'''
    Q = Q.T

    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    # Calculate error
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])

                    for k in range(K):
                        # Calculate the gradient with parameters [alpha, beta].
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])

        eR = numpy.dot(P,Q)

        e = 0

        for i in range(len(R)):

            for j in range(len(R[i])):

                if R[i][j] > 0:

                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)

                    for k in range(K):

                        e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        # e < 0.001 is considered a local minimum
        if e < 0.001:

            break

    return P, Q.T

# Example application with a toy user-items ratings matrix, with 5 items and 10 users. 
R = [

     [5,3,0,1,0],

     [4,0,0,1,0],

     [1,1,0,5,2],

     [1,0,0,4,1],

     [0,1,5,4,0],
    
     [2,1,3,0,5],
     
     [2,0,3,0,0],
     
     [0,1,3,0,0],

     [5,1,3,5,0],
     
     [3,1,3,0,3]

    ]

R = numpy.array(R)
# N: num of User
N = len(R)
# M: num of Movie
M = len(R[0])
# Num of Features
K = 3

P = numpy.random.rand(N,K)
Q = numpy.random.rand(M,K)

nP, nQ = matrix_factorization(R, P, Q, K)
nR = numpy.dot(nP, nQ.T)

# Output recommended user-items table. 
print(nR)