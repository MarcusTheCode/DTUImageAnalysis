# Image parameters
width, height = 1600, 800
bytes_per_pixel = 3  # RGB image (8-bits per channel)

# Calculate size per image in bytes
image_size_bytes = width * height * bytes_per_pixel

# Camera transfer rate
camera_fps = 6.25  # frames per second
data_transfer_per_sec_bytes = image_size_bytes * camera_fps  # bytes per second

# Convert to megabytes per second
data_transfer_per_sec_MB = data_transfer_per_sec_bytes / (1024 ** 2)

# Output
print(f"Each image size: {image_size_bytes / 1024:.2f} KB")
print(f"Data transferred per second: {data_transfer_per_sec_MB:.2f} MB/s")
