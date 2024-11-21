import matplotlib.pyplot as plt
from skimage import io, morphology, measure, transform,color

# Load the image
image = io.imread("dices3.jpg")

# Define the new resolution (height, width)
new_resolution = (900, 900)  # Example: 300px height, 400px width

# Resize the image
image = transform.resize(image, new_resolution, anti_aliasing=True)

# Convert the image to grayscale
#gray_image = color.rgb2gray(image)

# Threshold for white (values close to 1 in grayscale)
# Adjust the threshold value (e.g., 0.8) depending on your image
#white_mask = gray_image > 0.8
#cleaned_mask = morphology.binary_opening(white_mask, morphology.disk(100))
#cleaned_mask = morphology.remove_small_objects(cleaned_mask, min_size=1000)
#labeled_image = measure.label(cleaned_mask)  # Label connected components
#regions = measure.regionprops(labeled_image)

# Loop through the detected regions and draw bounding boxes
#for region in regions:
#    # Get the bounding box coordinates
##    minr, minc, maxr, maxc = region.bbox
    
#    # Draw the rectangle (bounding box)
#    rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr, 
#                         edgecolor='red', facecolor='none', linewidth=2)
#    ax.add_patch(rect)

# Show the result
#plt.title("Bounding Boxes Around Regions")
#plt.show()



# Extract the red channel
red_channel = image[:, :, 0]
green_channel = image[:, :, 1]
blue_channel = image[:, :, 2]


# Threshold the red channel to isolate red dice
red_mask = (red_channel > 40/255) & (green_channel < 110/255) & (blue_channel < 180/255)

#plt.imshow(red_mask)
#plt.axis("off")
#plt.title("Cropped Image")
#plt.show()
# Clean up the binary mask using morphological operations
#cleaned_mask = morphology.binary_opening(red_mask, morphology.disk(5))
cleaned_mask= red_mask
cleaned_mask = morphology.remove_small_objects(cleaned_mask, min_size=500)




#plt.imshow(cleaned_mask)
#plt.axis("off") 
#plt.title("binaryclosing Image")
#plt.show()
# Label connected components
# Label connected components
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

            # Optionally, resize the subimage to a target size (if you want consistent size)
            # subimage_resized = resize(subimage, (desired_size, desired_size), anti_aliasing=True)

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
    print(f"Subimage type: {type(subimage)}, dtype: {subimage.dtype}")

    # Convert subimage to grayscale
    gray_image = color.rgb2gray(subimage)

    # Threshold to create white mask
    white_mask = gray_image > 0.85

    # Label connected components in the white mask
    white_mask_morphed = morphology.area_opening(white_mask,area_threshold=20)
    labeled_image = measure.label(white_mask_morphed)
    
    regions = measure.regionprops(labeled_image)

    # Plot the subimage
    fig, ax = plt.subplots()
    ax.imshow(white_mask, cmap="gray")

    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        height = maxr - minr
        width = maxc - minc

        print(f"Region bounding box: ({minr}, {minc}, {maxr}, {maxc}), Width: {width}, Height: {height}")

        # Check if the bounding box dimensions are valid
        
        aspect_ratio = min(height, width) / max(height, width)

        if aspect_ratio > 0.6:
            print(f"Region bounding box: ({minr}, {minc}, {maxr}, {maxc}), Width: {width}, Height: {height}")
            rect = plt.Rectangle((minc, minr), width, height,
                                  edgecolor='red', facecolor='none', linewidth=2)
            ax.add_patch(rect)

    ax.axis("off")
    ax.set_title(f"Subimage {i+1} with Regions")
    plt.show()

