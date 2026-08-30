import cv2
import matplotlib.pyplot as plt

# Load image (grayscale)
img = cv2.imread('/content/story_lena_lenna_1.jpg', 0)  # Replace with your own image path
plt.figure(figsize=(5,5))
plt.title("Original Image")
plt.imshow(img, cmap='gray')
plt.axis('off')

# Manual Global Threshold
_, th_manual = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Otsu's Global Threshold
_, th_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plt.figure(figsize=(12,4))

plt.subplot(1,3,1); plt.imshow(img, cmap='gray'); plt.title("Original"); plt.axis('off')
plt.subplot(1,3,2); plt.imshow(th_manual, cmap='gray'); plt.title("Global Threshold (127)"); plt.axis('off')
plt.subplot(1,3,3); plt.imshow(th_otsu, cmap='gray'); plt.title("Otsu Threshold"); plt.axis('off')

plt.show()

# Canny with different thresholds
edges1 = cv2.Canny(img, 50, 150)
edges2 = cv2.Canny(img, 100, 200)

plt.figure(figsize=(10,4))
plt.subplot(1,3,1); plt.imshow(img, cmap='gray'); plt.title("Original"); plt.axis('off')
plt.subplot(1,3,2); plt.imshow(edges1, cmap='gray'); plt.title("Canny (50,150)"); plt.axis('off')
plt.subplot(1,3,3); plt.imshow(edges2, cmap='gray'); plt.title("Canny (100,200)"); plt.axis('off')

plt.show()
