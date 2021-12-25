"""
Vapnik/Chervonekis dimension examples
Code session 3 Tripods NSF REU-Graduate Stem for All 2021
"""

# import numpy
import numpy as np

# EXAMPLE 1
# Example 6.1 page 67 of Understanding Machine Learning: From Theory to Algorithms by Shai Shalev-Shwartz and Shai Ben-David

# List of pairs of the form (x,1) or (x,-1) for x in the appropriate range is created
S = np.array([[1,1],[-10,1],[100,1],[1000,-1],[500,-1]])

# Sets initial bounds S_inf and S_sup, which are negative and positive infinity.
S_inf = -np.Inf
S_sup = np.Inf

# Loop finds the smallest input in the sample such that the label is -1 and the largest input such that the label is 1:
size = 5
for i in range(size):
    if S[i,0] > S_inf:
        if S[i,1] == 1:
            S_inf = S[i,0]
        else:
            if S[i,0] < S_sup:
                S_sup = S[i,0]

# The learned threshold function is the indicator function for values up to the average of S_inf and S_sup.
threshold = (S_inf + S_sup)/2
# Print the treshold
print(threshold)

#EXAMPLE 2
#Use od halfspaces and linear programming to classify points in R^2
#ERM rule is implemented exploiting linear programming as explained in 9.1.1 at pg.119 of Understanding Machine Learning: From Theory to Algorithms by Shai Shalev-Shwartz and Shai Ben-David

# Imports linear program solver
from scipy.optimize import linprog

# Collection of 3 points in R^2 (including label of binary classification in last coordinate)constituting the training set
S = np.array([[1,2,1],[2,1,-1],[4,5,1]])
size = 3

# Creates a vector u consisting of as many zeroes as there are coordinates in our points.
u = np.zeros(2)
# Create an array S_x which consists of only the points, not their labels.
S_x = S[0:size,0:2]
# Makes array S_label which consisting of only the labels of the training set.
S_label = np.reshape(S[0:size,2],(3,1))
# Multiplication to negate the appropriate rows so that linear programming will max/minimize as needed.
A = S_x * S_label
# Negates the resulting array or to ensure that the learned function has labels with appropriate sign.
B = -A
# Makes an array consisting of all -1 whose length is the number of coordinates in a point.
v = -np.ones(size)
# Uses linprog to solve the linear program and display the solution.
w = linprog(u, A_ub=B, b_ub=v, bounds=((None, None),(None, None)))
print(w)

# REAL WORLD EXAMPLE
# Use of classifiers on real-world data sets.

# imports pandas and gets csv files
import pandas as pd
train_data = pd.read_csv('fashion-mnist_train.csv')
test_data = pd.read_csv('fashion-mnist_test.csv')

# Relabels the data with the first column 'Label' and the latter columns
# (1,1) through (28,28), which represent the pixels in a 28 by 28 greyscale image of a clothing article.
train_data.columns = ['Label']+[(i,j) for i in range(1,29) for j in range(1,29)]
test_data.columns = ['Label']+[(i,j) for i in range(1,29) for j in range(1,29)]

# Keep only the tops (0) and pants (1) labels using the data.loc method for both datasets.
train_data = train_data.loc[(train_data['Label']==0) | (train_data['Label']==1)]
test_data = test_data.loc[(test_data['Label']==0) | (test_data['Label']==1)]

# Convert the data to a numpy array and change the zero labels to -1 so that the same linear solver technique from before will work.
train_np = train_data.to_numpy()
for i in range(len(train_np)):
    if train_np[i][0] == 0:
        train_np[i][0] = -1
test_np = test_data.to_numpy()
for i in range(len(test_np)):
    if test_np[i][0] == 0:
        test_np[i][0] = -1

import matplotlib.pyplot as plt
print(train_np.shape)
train_images = train_np[:,1:].reshape(12000,28,28)
print(train_images.shape)

# Plots trining set with clothes and labels
plt.figure(figsize=(10,10))
for i in range(25):
    
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(train_np[i,0])
plt.show()

# Takes 500 samples from the training dataset and use the linear solver on them.
# Points are in a 28**2 dimensional space instead of the 2 dimensional plane R^2.
size = 500
assert size<len(train_np)
u = np.zeros(28**2)
S_x = train_np[0:size,1:28**2+1]
S_label = np.reshape(train_np[0:size,0],(size,1))
A = S_x * S_label
B = -A 
v = -np.ones(size)
# Solves linear program and print solution
w = linprog(u, A_ub=B, b_ub=v, bounds=tuple((None,None) for i in range(28**2)))
print(w)

# displays it graphically
print(w.x.shape)
result = w.x.reshape(28,28)
print(result.shape)
res_img= np.interp(result,[np.min(result),np.max(result)],[0,255])
plt.imshow(res_img)
plt.colorbar()
plt.show()

# Computes the fraction of successes on the test set through the dot product test
successes = 0
for i in range(size):
    if np.dot(w.x,test_np[i,1:785])*test_np[i,0]>0:
        successes += 1
print(successes/size)













