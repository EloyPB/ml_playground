import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


iris = load_iris()
X = iris.data


# #  PLOT DATASET

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# for target_num, target_name in enumerate(iris.target_names):
#     ax.scatter(X[iris.target == target_num, 0], X[iris.target == target_num, 1], 
#                X[iris.target == target_num, 2], label=target_name)
    
# ax.legend()


#  LEARN 

num_components = 3
num_points = X.shape[0]

#  initialization

means = X[np.random.choice(num_points, 3, replace=False)].T
mean = np.mean(X, 0)

cov = (X - mean).T @ (X - mean) / (num_points - 1)
cov = np.repeat(cov[:, :, np.newaxis], num_components, axis=2)

mixing_weights = np.ones(num_components) / num_components

log_likelihoods = []

# expectation maximization

for i in range(1000):

    # 1) E-step (expected value of the latent random variables: posterior probability of each component for each point)
    
    # probability of each point under each component (likelihood in Baye's rule sense)
    likelihood = np.empty((num_points, num_components))
    
    for point_num in range(num_points):
        for component_num in range(num_components):
            diff = (X[point_num] - means[:, component_num])
            exponent = -0.5 * diff @ np.linalg.inv(cov[:, :, component_num]) @ diff.T
            norm_factor = np.sqrt((2 * np.pi) ** X.shape[1] * np.linalg.det(cov[:, :, component_num]))
            likelihood[point_num, component_num] = np.exp(exponent) / norm_factor
            
    # compute log-likelihood of the entire dataset under the mixture model
    log_likelihood = np.sum(np.log(np.sum(mixing_weights * likelihood, axis=1)))
    log_likelihoods.append(log_likelihood)

    # check convergence
    if i > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < 1e-6:
        print(f"Converged at iteration {i}")
        break
        
            
    # probality each poing belongs to each component (posterior probability; using Bayes' rule)
    p_components = mixing_weights * likelihood;
    p_components = p_components / np.sum(p_components, axis=1, keepdims=True)
    
    

    
    # 2) M-step (maximize expected log-likelihood with respect to parameters)
    
    # update mixing weights
    component_sums = np.sum(p_components, 0)
    mixing_weights = component_sums / num_points
    
    # update means
    means = X.T @ p_components / component_sums
    
    
    # update covariance matrices
    for component_num in range(num_components):
        cov[:, :, component_num] = (p_components[:, component_num] * 
                                    (X - means[:, component_num]).T @ (X - means[:, component_num]) / component_sums[component_num])
        


# PLOT CLASSIFICATION

assignments = np.argmax(p_components, axis=1)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=assignments, cmap='brg')
    




