import numpy as np

def LDA(X, y):
    """
    Linear Discriminant Analysis with equal priors and shared covariance.
    """
    n, m = X.shape
    class_label = np.unique(y)
    k = len(class_label)

    n_group     = np.zeros((k,1))
    group_mean  = np.zeros((k,m))
    pooled_cov  = np.zeros((m,m))
    W           = np.zeros((k,m+1))

    for i in range(k):
        group       = np.squeeze(y == class_label[i])
        n_group[i]  = np.sum(group.astype(np.double))
        group_mean[i,:] = np.mean(X[group,:], axis=0)
        pooled_cov += ((n_group[i] - 1) / (n - k)) * np.cov(X[group,:], rowvar=False)

    prior_prob = n_group / n

    for i in range(k):
        temp = group_mean[i,:][np.newaxis] @ np.linalg.inv(pooled_cov)
        W[i,0] = -0.5 * temp @ group_mean[i,:].T + np.log(prior_prob[i])
        W[i,1:] = temp

    return W

# Manual LDA computation since only one sample per class leads to NaN covariance
mu1 = np.array([24, 3])
mu2 = np.array([30, 7])
cov_matrix = np.array([[2, 0], [0, 2]])
x = np.array([23, 5])

cov_inv = np.linalg.inv(cov_matrix)
w = cov_inv @ (mu2 - mu1)
c_w = 0.5 * (mu2 + mu1).T @ cov_inv @ (mu2 - mu1)
y_class2 = x @ w
predicted_class = 2 if y_class2 > c_w else 1

print(f"y_class2(x): {y_class2}")
print(f"Threshold (c_w): {c_w}")
print(f"Predicted class: {predicted_class}")
