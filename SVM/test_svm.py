import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass, shift
from skimage.filters import threshold_otsu
from skimage.io import imread
from skimage.color import rgb2gray, rgba2rgb
from skimage.transform import resize
from skimage.feature import hog
import joblib

model = joblib.load("./svm_hog_model.joblib")
scaler = joblib.load("./scaler.joblib")

# Load image and preprocess
img_path = "../test_images/30_00005.png"
img = imread(img_path)

# Grayscale
if img.ndim == 3:
    if img.shape[2] == 4:
        img = rgba2rgb(img)
    img = rgb2gray(img)

# Normalize 0-1
img = (img - img.min()) / (img.max() - img.min() + 1e-8)

# Threshold (Otsu)
thresh = threshold_otsu(img)
img = img > thresh
img = img.astype(np.float32)

# Ensure digit is white
if np.mean(img) > 0.5:
    img = 1 - img

# Bounding box crop
coords = np.column_stack(np.where(img > 0))
y_min, x_min = coords.min(axis=0)
y_max, x_max = coords.max(axis=0)
digit = img[y_min:y_max+1, x_min:x_max+1]

# Resize
digit = resize(digit, (20, 20), anti_aliasing=True)
canvas = np.zeros((28, 28))
canvas[4:24, 4:24] = digit

# Center shift (center of mass)
cy, cx = center_of_mass(canvas)
shift_y = 14 - cy
shift_x = 14 - cx
canvas = shift(canvas, (shift_y, shift_x))

# HOG
features = hog(
    canvas,
    orientations=9,
    pixels_per_cell=(4, 4),
    cells_per_block=(2, 2),
    block_norm='L2-Hys'
)
hog_features = features.reshape(1, -1)
hog_scaled = scaler.transform(hog_features)

# Predict
prediction = model.predict(hog_scaled)
print("Predicted digit:", prediction[0])

# Show image
plt.imshow(canvas, cmap='gray')
plt.title(f"Predicted: {prediction[0]}")
plt.axis("off")
plt.show()