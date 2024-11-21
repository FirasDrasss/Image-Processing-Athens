import matplotlib.pyplot as plt
from skimage import io, morphology, measure, transform,color
import numpy as np

# Load the image
image = io.imread("dices9.jpg")

# Define the new resolution (height, width)
new_resolution = (900, 900)  # Example: 300px height, 400px width

# Resize the image
image = transform.resize(image, new_resolution, anti_aliasing=True)

# Extract the red channel
red_channel = image[:, :, 0]
green_channel = image[:, :, 1]
blue_channel = image[:, :, 2]


# Threshold the red channel to isolate red dice
red_mask = (red_channel > 40/255) & (green_channel < 110/255) & (blue_channel < 180/255)

# Clean up the binary mask using morphological operations
cleaned_mask= red_mask
cleaned_mask = morphology.remove_small_objects(cleaned_mask, min_size=500)



labeled_image = measure.label(cleaned_mask)  # Label connected components
regions = measure.regionprops(labeled_image)

# Define the desired size range for the bounding box (min_area, max_area)
min_area = 1000  # Minimum area in pixels (height * width)
max_area = 10000  # Maximum area in pixels

# Initialize a list to store the subimages
subimages = []

# Loop through the detected regions and extract square-like regions within the size range
for region in regions:
    
    # Get the bounding box coordinates (minr, minc, maxr, maxc)
    minr, minc, maxr, maxc = region.bbox
    
    # Calculate the width and height of the bounding box
    height = maxr - minr
    width = maxc - minc

    # Check if the region is square-like (aspect ratio close to 1)
    aspect_ratio = min(height, width) / max(height, width)
    if aspect_ratio > 0.75:  # Aspect ratio threshold for square-like regions
        area = height * width
        # Check if the area is within the specified range
        if min_area <= area <= max_area:
            # Crop the region from the original image (can use either color or binary mask)
            subimage = image[minr:maxr, minc:maxc]

            # Add the cropped subimage to the list
            subimages.append(subimage)

# Display the extracted subimages
if subimages:
    fig, ax = plt.subplots(1, len(subimages), figsize=(15, 5))
    for i, subimage in enumerate(subimages):
        ax[i].imshow(subimage)
        ax[i].axis("off")
        ax[i].set_title(f"Dice {i+1}")
    plt.tight_layout()
    plt.show()
else:
    print("No square-like regions found that meet the size criteria.")

for i, subimage in enumerate(subimages):
    print(f"Processing Subimage {i+1}")
    
    new_resolution = (400, 400)  # Example: 300px height, 400px width

    # Resize the image
    subimage = transform.resize(subimage, new_resolution, anti_aliasing=True)
    # Convert subimage to grayscale
    gray_image = color.rgb2gray(subimage)

    white_mask = gray_image > 0.75
    # Apply closing to fill small gaps in the dots
    margin = 65
    white_mask[:margin, :] = False  # Top edge
    white_mask[:, -margin:] = False  # Right edge
    # Label connected components in the white mask
    white_mask_morphed = morphology.erosion(white_mask, morphology.disk(3))
    white_mask_morphed = morphology.area_opening(white_mask_morphed,area_threshold=1200)
    labeled_image = measure.label(white_mask_morphed)
    
    regions = measure.regionprops(labeled_image)

    # Plot the subimage
    fig, ax = plt.subplots()
    ax.imshow(labeled_image, cmap="gray")

    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        height = maxr - minr
        width = maxc - minc
        area = region.area
        perimeter = region.perimeter
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        

        # Check if the bounding box dimensions are valid
        
        aspect_ratio = min(height, width) / max(height, width)

        if aspect_ratio > 0.5 and circularity > 0.4:
            print(f"Region bounding box: ({minr}, {minc}, {maxr}, {maxc}), Width: {width}, Height: {height}")
            rect = plt.Rectangle((minc, minr), width, height,
                                    edgecolor='red', facecolor='none', linewidth=2)
            ax.add_patch(rect)

    ax.axis("off")
    ax.set_title(f"Subimage {i+1} with Regions")
    plt.show()

