import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, thin

# Create a simple shape image
img = np.zeros((300,300), dtype="uint8")
cv2.rectangle(img, (50,50), (250,250), 255, -1)
cv2.circle(img, (150,150), 70, 0, -1)

plt.figure(figsize=(5,5))
plt.title("Original Shape")
plt.imshow(img, cmap='gray')
plt.axis('off')


# Convert to binary
bin_img = img.copy()
bin_img[bin_img > 0] = 1

# Skeletonization using skimage
skeleton = skeletonize(bin_img)

plt.figure(figsize=(5,5))
plt.title("Skeletonization")
plt.imshow(skeleton, cmap='gray')
plt.axis('off')


thinned = thin(bin_img)

plt.figure(figsize=(5,5))
plt.title("Thinning (Zhang–Suen)")
plt.imshow(thinned, cmap='gray')
plt.axis('off')


import urllib.request

# Download a sample image with edges
url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/sudoku.png"
img_path = "sudoku.png"
urllib.request.urlretrieve(url, img_path)

img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)

plt.figure(figsize=(6,6))
plt.title("Edges")
plt.imshow(edges, cmap='gray')
plt.axis('off')


lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=120,
                        minLineLength=50, maxLineGap=10)

line_img = img.copy()
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(line_img, (x1,y1), (x2,y2), (0,0,255), 2)

plt.figure(figsize=(6,6))
plt.title("Detected Lines (HoughLinesP)")
plt.imshow(cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB))
plt.axis('off')


from google.colab import files
import cv2
import numpy as np
from matplotlib import pyplot as plt

# ---- Upload image ----
print("Upload an image for CIRCLE detection (e.g., coins.png):")
uploaded2 = files.upload()

circle_img_path = list(uploaded2.keys())[0]
print("Using:", circle_img_path)

# ---- Read image ----
img2 = cv2.imread(circle_img_path)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray2 = cv2.medianBlur(gray2, 5)

# ---- Hough Circle Detection ----
circles = cv2.HoughCircles(
    gray2,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=40,
    param1=100,
    param2=30,
    minRadius=10,
    maxRadius=80
)

output = img2.copy()

# ---- Draw circles ----
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv2.circle(output, (x, y), r, (0, 255, 0), 2)
        cv2.circle(output, (x, y), 2, (255, 0, 0), 3)
else:
    print("No circles found!")

# ---- Display ----
plt.figure(figsize=(10,5))
plt.subplot(1,2,1); plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)); plt.title("Original")
plt.subplot(1,2,2); plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB)); plt.title("Detected Circles")
plt.show()
