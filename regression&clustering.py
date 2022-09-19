# Imports
from turtle import st
import numpy as np
from sklearn.cluster import kmeans_plusplus, KMeans
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

#set random seed
np.random.seed(123)

#Load test and train data
train_data = loadmat("./sarcos_inv.mat")["sarcos_inv"]
test_data = loadmat("./sarcos_inv_test.mat")["sarcos_inv_test"]

#validation split
X_train, X_val, y_Train, y_Val = train_test_split(train_data[:,:21],train_data[:,21])
y_train, y_val = y_Train.reshape((-1,1)),y_Val.reshape((-1,1))
X_test, y_test = test_data[:,:21], test_data[:,21].reshape((-1,1))

# Normalize the data
class Standardize():
    def __init__(self) -> None:
        pass
        
        
    # Standardize data
    def normalize(self,data):
        data_std = data.copy()
        mu = np.mean(data_std)
        sigma = np.std(data_std)

        return (data_std - mu)/sigma

# Store standardized data into new variables
std = Standardize()

X_train_std = std.normalize(X_train)
X_val_std = std.normalize(X_val)
y_train_std = std.normalize(y_train)
y_val_std = std.normalize(y_val)

X_test_std = std.normalize(X_test)
y_test_std = std.normalize(y_test)

# Variance calculation
def variance(xs):
    return float(np.sum(np.square(xs-xs.mean()))/(len(xs)))

# Standardized Mean Squared Error (SMSE)

def smse(z1,z2,s):
    """ z1, z2 are data values of equal length and s is the normalization factor"""
    return float((sum(np.square(z2-z1)))/len(z1))

# Linear Regression
""" Solve system of linear equations y = wTx and find the weights"""
Weights = np.linalg.pinv(X_train_std)@ y_train_std

# Predict the validation results using the weights
y_pred_valid = X_val_std @ Weights

# Find the error in the prediction

SMSE = smse(y_pred_valid,y_val_std,variance(y_train))

# The result is around 0.076 and now we try to improve it using poly features upto degree 3
poly_feat = PolynomialFeatures(degree=3)
x_train_poly = poly_feat.fit_transform(X_train_std)
x_val_poly = poly_feat.fit_transform(X_val_std)

# Recalculate the weights
Weights_poly = np.linalg.pinv(x_train_poly) @ y_train_std

# re-estimate the y_validation
y_pred_val_poly = x_val_poly @ Weights_poly

# Find the error in the prediction

SMSE = smse(y_pred_val_poly,y_val_std,variance(y_train))

# The error gets more than halved. New error = 0.032

""" Create a Radial Basis Fuction Network using K-Means for regression"""

# Use 100 cluster center for kmeans++

kmeans_clust = KMeans(n_clusters=100, init="k-means++", random_state=64).fit(X_train_std)
x_centroids = kmeans_clust.cluster_centers_

# Implement the Gaussian basis funcs and transform data
def RBF_network(xs, centers):
    
    N = len(xs)
    C = len(centers)
    
    xs_gauss = []
    
    # calculate euclidean distance of two points
    def get_distance(x, c):
        return np.sqrt(np.sum(np.square(x - c)))
    
    # Radial Basis Functions
    def RBF(x, centroids, std=25):
        r = get_distance(x, centroids)
        denominator = np.sqrt(2*np.pi*std**2)
        nominator = np.exp((-r**2)/(2*std**2))
        return nominator / denominator
    
    for i in range(N):
        RBF_array = []
        
        # include bias term
        RBF_array.insert(0,1)
        
        for j in range(C):
            temp_rbf = RBF(xs[i], centers[j])
            RBF_array.append(temp_rbf)
        
        xs_gauss.append(RBF_array)
        
    
    return np.array(xs_gauss, dtype=np.float64)

xs_train_gauss = RBF_network(X_train_std, x_centroids)

# Use regression on the transformed data

w_rbf = np.linalg.pinv(xs_train_gauss) @ y_train_std
xs_valid_gauss = RBF_network(X_val_std, x_centroids)

# This should contain the resulting predictions on the validation data set
ys_pred_gauss_valid = xs_valid_gauss @ w_rbf

# This should contain the corresponding SMSE
smse_gauss = smse(ys_pred_gauss_valid, y_val_std, variance(y_train))

# we get an error of 0.047 with linear regression
