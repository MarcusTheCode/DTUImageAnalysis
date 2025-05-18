import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt


# Load data
train_data = np.loadtxt("traffic_train.txt", delimiter=',')
test_data = np.loadtxt("traffic_test.txt", delimiter=',')

# Extract features (density and speed) and labels
X_train = train_data[:, :2]
y_train = np.array([0]*140 + [1]*140)  # 0: morning, 1: afternoon

X_test = test_data[:, :2]
y_test_true = np.array([0]*60 + [1]*60)

# Train LDA model
model = LDA()
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Count afternoon samples classified as morning (traffic jam)
afternoon_misclassified = np.sum(y_pred[60:] == 0)
print("Afternoon samples classified as morning:", afternoon_misclassified)

# Extract the weather column for morning data (first 140 rows)
morning_weather = train_data[:140, 2]

# Count days with rain (weather == 1)
rainy_morning_days = np.sum(morning_weather == 1)

print("Number of rainy mornings in training set:", int(rainy_morning_days))

# Extract features
X = train_data[:, :2]  # density and speed

# Separate the first 100 samples for each class
morning_samples = X[:100]
afternoon_samples = X[140:240]

# Plot the samples
plt.figure(figsize=(8, 6))
plt.scatter(morning_samples[:, 0], morning_samples[:, 1], color='green', label='Morning')
plt.scatter(afternoon_samples[:, 0], afternoon_samples[:, 1], color='blue', label='Afternoon')
plt.xlabel('Density')
plt.ylabel('Speed')
plt.title('Scatter plot of Density vs Speed')
plt.legend()
plt.grid(True)
plt.show()