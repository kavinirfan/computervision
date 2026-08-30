!pip install opencv-contrib-python
import cv2
import matplotlib.pyplot as plt

# Download a suitable image for feature extraction
import urllib.request
url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/building.jpg"
img_path = "building.jpg"
urllib.request.urlretrieve(url, img_path)

img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(6,6))
plt.imshow(gray, cmap='gray')
plt.title("Input Image")
plt.axis('off')


sift = cv2.SIFT_create()
kp_sift, des_sift = sift.detectAndCompute(gray, None)

img_sift = cv2.drawKeypoints(gray, kp_sift, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.figure(figsize=(6,6))
plt.imshow(img_sift, cmap='gray')
plt.title("SIFT Keypoints")
plt.axis('off')


akaze = cv2.AKAZE_create()
kp_akaze, des_akaze = akaze.detectAndCompute(gray, None)

img_akaze = cv2.drawKeypoints(gray, kp_akaze, None,
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.figure(figsize=(6,6))
plt.imshow(img_akaze, cmap='gray')
plt.title("AKAZE Keypoints (SURF Alternative)")
plt.axis('off')
