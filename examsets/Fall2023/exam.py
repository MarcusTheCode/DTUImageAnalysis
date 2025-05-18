import numpy as np
import cv2
import matplotlib.pyplot as plt

# Create a blank image
img = np.zeros((200, 200), dtype=np.uint8)

# Scale to convert line rho values into pixels
# Assume each unit ~50 pixels for clarity
center_x, center_y = 100, 100

# Body (vertical line, theta = 0°, rho ≈ 3)
cv2.line(img, (center_x + 50, center_y - 50), (center_x + 50, center_y + 50), 255, 2)

# Arms
# Left arm, theta = -45°, rho ≈ -0.7
cv2.line(img, (center_x + 50, center_y - 20), (center_x + 20, center_y + 10), 255, 2)
# Right arm, theta = 45°, rho ≈ 3.5
cv2.line(img, (center_x + 50, center_y - 20), (center_x + 80, center_y + 10), 255, 2)

# Legs
# Left leg, theta = -45°, rho ≈ 0.7
cv2.line(img, (center_x + 50, center_y + 50), (center_x + 20, center_y + 80), 255, 2)
# Right leg, theta = 45°, rho ≈ 4.9
cv2.line(img, (center_x + 50, center_y + 50), (center_x + 80, center_y + 80), 255, 2)

# Apply Hough Transform
lines = cv2.HoughLines(img, 1, np.pi / 180, 80)

# Display detected lines
theta_rho = []
if lines is not None:
    for line in lines:
        rho, theta = line[0]
        theta_deg = np.rad2deg(theta) - 90  # Convert to ]-90, 90]
        theta_deg = (theta_deg + 180) % 180 - 90  # Normalize to ]-90, 90]
        theta_rho.append((round(theta_deg), round(rho / 50.0, 1)))  # Normalize rho back to ~unit scale

# Sort and print results
theta_rho = list(set(theta_rho))  # Remove duplicates
theta_rho.sort()
print("Detected lines (Theta, Rho):")
for theta, rho in theta_rho:
    print(f"({theta}, {rho})")

# Show the image
plt.imshow(img, cmap='gray')
plt.title("Matchstick Person")
plt.axis('off')
plt.show()
