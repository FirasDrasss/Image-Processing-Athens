import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, morphology, measure

# Load the image
image = io.imread("dices9.jpg")

# Extract the red channel
red_channel = image[:, :, 0]
green_channel = image[:, :, 1]
blue_channel = image[:, :, 2]
"""
# Plot the original image and the segmentation result
fig, ax = plt.subplots(1, 3, figsize=(12, 6))

# Original image
ax[0].imshow(red_channel)
ax[0].set_title("Original Image")
ax[0].axis("off")

# Segmented dice
ax[1].imshow(blue_channel)
ax[1].set_title("Segmented Dice")
ax[1].axis("off")

ax[2].imshow(green_channel)
ax[2].set_title("Segmented Dice")
ax[2].axis("off")


plt.tight_layout()
plt.show()
"""
# Threshold the red channel to isolate red dice
red_mask = (red_channel > 100) & (green_channel < 100) & (blue_channel < 100)

plt.imshow(red_mask)
plt.axis("off")
plt.title("Cropped Image")
plt.show()
# Clean up the binary mask using morphological operations
cleaned_mask = morphology.binary_closing(red_mask, morphology.disk(5))
cleaned_mask = morphology.remove_small_objects(cleaned_mask, min_size=5000)


plt.imshow(cleaned_mask)
plt.axis("off")
plt.title("binaryclosing Image")
plt.show()
# Label connected components
labeled_image = measure.label(cleaned_mask)  # Label connected components
regions = measure.regionprops(labeled_image)

fig, ax = plt.subplots()
ax.imshow(cleaned_mask, cmap=plt.cm.gray)
for props in regions:
    y0, x0 = props.centroid
    orientation = props.orientation
    minr, minc, maxr, maxc = props.bbox
    bx = (minc, maxc, maxc, minc, minc)
    by = (minr, minr, maxr, maxr, minr)
    ax.plot(bx, by, '-b', linewidth=2.5)

ax.axis((0, 600, 600, 0))
plt.show()

#plt.imshow(filtered)
#plt.axis("off")
#plt.title("filtered Image")
#plt.show()
# Plot the original image and the segmentation result
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
"""
# Original image
ax[0].imshow(image)
ax[0].set_title("Original Image")
ax[0].axis("off")

# Segmented dice
ax[1].imshow(color.label2rgb(labeled_dice, image=image, bg_label=0))
ax[1].set_title("Segmented Dice")
ax[1].axis("off")

plt.tight_layout()
plt.show()
"""