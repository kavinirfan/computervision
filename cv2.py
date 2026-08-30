import cv2
import matplotlib.pyplot as plt

from google.colab import files
uploaded = files.upload()

img_path = list(uploaded.keys())[0]
img = cv2.imread(img_path, 0)

plt.figure(figsize=(5,5))
plt.title("Real-World Image")
plt.imshow(img, cmap='gray')
plt.axis('off')

sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel_mag = cv2.magnitude(sobel_x, sobel_y)
laplace = cv2.Laplacian(img, cv2.CV_64F)
canny = cv2.Canny(img, 100, 200)

plt.figure(figsize=(12,8))

plt.subplot(2,2,1); plt.imshow(sobel_mag, cmap='gray'); plt.title("Sobel Magnitude"); plt.axis('off')
plt.subplot(2,2,2); plt.imshow(laplace, cmap='gray'); plt.title("Laplacian"); plt.axis('off')
plt.subplot(2,2,3); plt.imshow(canny, cmap='gray'); plt.title("Canny"); plt.axis('off')
plt.subplot(2,2,4); plt.imshow(img, cmap='gray'); plt.title("Original"); plt.axis('off')

plt.show()
