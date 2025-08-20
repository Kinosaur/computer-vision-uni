import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def hough_line_detection():
    # Read the image
    img = cv.imread('week6/1.jpg')
    if img is None:
        print("Error: Could not read image. Please check the file path.")
        return
    
    # Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    
    # Make copies for different methods
    img_standard = img.copy()
    img_probabilistic = img.copy()
    
    # Method 1: Standard Hough Line Transform
    lines_standard = cv.HoughLines(edges, 1, np.pi/180, threshold=200)
    
    if lines_standard is not None:
        for line in lines_standard:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv.line(img_standard, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Method 2: Probabilistic Hough Line Transform
    lines_prob = cv.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                                minLineLength=100, maxLineGap=10)
    
    if lines_prob is not None:
        for line in lines_prob:
            x1, y1, x2, y2 = line[0]
            cv.line(img_probabilistic, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Display results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(gray, cmap='gray')
    plt.title('Grayscale')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(edges, cmap='gray')
    plt.title('Edge Detection (Canny)')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(cv.cvtColor(img_standard, cv.COLOR_BGR2RGB))
    plt.title('Standard Hough Lines')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(cv.cvtColor(img_probabilistic, cv.COLOR_BGR2RGB))
    plt.title('Probabilistic Hough Lines')
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    # Show both methods combined
    img_combined = img.copy()
    if lines_standard is not None:
        for line in lines_standard:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv.line(img_combined, (x1, y1), (x2, y2), (0, 0, 255), 1)
    
    if lines_prob is not None:
        for line in lines_prob:
            x1, y1, x2, y2 = line[0]
            cv.line(img_combined, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    plt.imshow(cv.cvtColor(img_combined, cv.COLOR_BGR2RGB))
    plt.title('Combined (Red: Standard, Green: Probabilistic)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"Standard Hough Lines detected: {len(lines_standard) if lines_standard is not None else 0}")
    print(f"Probabilistic Hough Lines detected: {len(lines_prob) if lines_prob is not None else 0}")

if __name__ == "__main__":
    hough_line_detection()