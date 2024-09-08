from PIL import Image
import cv2

def detect_blank_patch(image_array, threshold=3.6):
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    mean, std_dev = cv2.meanStdDev(laplacian)
    return mean[0][0], std_dev[0][0], std_dev[0][0]<threshold

