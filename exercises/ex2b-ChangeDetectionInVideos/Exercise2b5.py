import time
import cv2
import numpy as np
from skimage.util import img_as_float, img_as_ubyte

def show_in_moved_window(win_name, img, x, y):
    """
    Show an image in a window, where the position of the window can be given
    """
    cv2.namedWindow(win_name)
    cv2.moveWindow(win_name, x, y)
    cv2.imshow(win_name, img)

def capture_from_camera_and_show_images():
    print("Starting image capture")

    print("Opening connection to camera")
    url = 0
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    print("Acquiring background image")
    ret, background = cap.read()
    if not ret:
        print("Can't receive frame")
        exit()
    
    background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    background_gray = img_as_float(background_gray)
    
    alpha = 0.2  # Weight for background update
    alert_threshold = 0.05  # Threshold for change detection
    
    start_time = time.time()
    n_frames = 0
    stop = False
    
    while not stop:
        ret, new_frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting ...")
            break
        
        new_frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        new_frame_gray = img_as_float(new_frame_gray)
        
        # Compute absolute difference image
        dif_img = np.abs(new_frame_gray - background_gray)
        
        # Apply threshold to create binary image
        threshold = 0.1
        binary_img = (dif_img > threshold).astype(np.uint8)
        binary_img = img_as_ubyte(binary_img)
        
        # Count foreground pixels
        F = np.sum(binary_img > 0)
        total_pixels = binary_img.size
        foreground_percentage = F / total_pixels
        # Compute statistics on difference image
        diff_min = np.min(dif_img)
        diff_max = np.max(dif_img)
        diff_avg = np.mean(dif_img)
        
        # Display information on the input image
        cv2.putText(new_frame, f"Changed Pixels: {F}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(new_frame, f"Diff Min: {diff_min:.3f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(new_frame, f"Diff Max: {diff_max:.3f}", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(new_frame, f"Diff Avg: {diff_avg:.3f}", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Check if alarm should be raised
        if foreground_percentage > alert_threshold:
            cv2.putText(new_frame, "Change Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display images
        show_in_moved_window('Input', new_frame, 0, 10)
        show_in_moved_window('Background', background_gray, 600, 10)
        show_in_moved_window('Difference Image', dif_img, 1200, 10)
        show_in_moved_window('Binary Image', binary_img, 10, 800)
        
        # Update background image
        background_gray = alpha * background_gray + (1 - alpha) * new_frame_gray
        
        # Track FPS
        n_frames += 1
        elapsed_time = time.time() - start_time
        fps = int(n_frames / elapsed_time)
        cv2.putText(new_frame, f"FPS: {fps}", (100, 180), cv2.FONT_HERSHEY_COMPLEX, 1, 255, 1)
        
        if cv2.waitKey(1) == ord('q'):
            stop = True
    
    print("Stopping image loop")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    capture_from_camera_and_show_images()