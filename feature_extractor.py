import cv2
import csv
import numpy as np
import pandas as pd
from skimage.segmentation import chan_vese
from skimage import exposure
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, remove_small_objects
from skimage.filters import median, threshold_otsu, threshold_sauvola
from skimage.color import rgb2gray
from skimage import img_as_float

def remove_hairs(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Define the kernel (structural element)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

    # 1. Black Hat Transform (Detects Dark Hairs)
    # Black Hat = Closing - Original
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, blackhat_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
    # 2. White Hat Transform (Detects Bright Hairs)
    # White Hat (Top Hat) = Original - Opening
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    _, tophat_mask = cv2.threshold(tophat, 30, 255, cv2.THRESH_BINARY) 
    #     # The two transforms work in opposition: Black Hat finds dark lines, White Hat finds bright lines.

    # 3. Combine both masks using OR logic
    # The combined mask will now contain all dark *and* bright hair pixels
    combined_mask = cv2.bitwise_or(blackhat_mask, tophat_mask)

    # 4. Inpaint using the comprehensive mask
    result = cv2.inpaint(img, combined_mask, 3, cv2.INPAINT_TELEA)
    return result


def contour_sauvola(cleaned_img):
    # Use green channel and convert to grayscale
    img_gray = cleaned_img[:, :, 1]
    
    # Window size determines are for threshold. Must be odd.
    # K controls sensitivity.
    window_size = 405 # Must be an odd number
    k_sensitivity = 0.1
    
    thresh_sauvola = threshold_sauvola(img_gray, window_size=window_size, k=k_sensitivity)
    
    # Create a mask from pixels darker than the local threshold
    binary_mask = img_gray < thresh_sauvola
    """
    # Create a circular mask to ignore the vignette corners
    h, w = binary_mask.shape
    center = (int(w/2), int(h/2))
    radius = int(min(h, w) / 1.5) - 10 
    
    circle_mask = np.zeros_like(binary_mask, dtype=np.uint8)
    cv2.circle(circle_mask, center, radius, 1, thickness=-1)
    
    # Apply the circle mask to your binary mask
    binary_mask = np.logical_and(binary_mask, circle_mask)
    """

    # Fill small holes and smoothe the boundary
    closed_mask = closing(binary_mask, square(5)) 
    
    # Remove small objects
    labeled_mask = label(closed_mask)
    final_mask = remove_small_objects(labeled_mask, min_size=5000)
    
    return final_mask


def compute_asymmetry(label_img):
    
    #M = cv2.moments(largest_contour)
    #cx = int(M['m10'] / M['m00'])
    #cy = int(M['m01'] / M['m00'])
    #bin_mask = label_img > 0
    largest_contour = label(label_img)
    props = regionprops(largest_contour)[0]
    cy, cx = props.centroid
    orientation = props.orientation
    
    M = cv2.getRotationMatrix2D((cx, cy), -np.degrees(orientation), 1.0)
    rotated = cv2.warpAffine(largest_contour.astype(np.uint8), M, (label_img.shape[1], label_img.shape[0]))
    
    flipped_x = cv2.flip(rotated, 1)  # horizontal
    
    delta_x = np.logical_xor(rotated, flipped_x).sum()
    AS1 = delta_x / rotated.sum()

    flipped_y = cv2.flip(rotated, 0)  # vertical
    delta_y = np.logical_xor(rotated, flipped_y).sum()
    AS2 = delta_y / rotated.sum()

    return AS1, AS2  

def compute_border_features(mask, contour):

    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # BI2: Perimeter to Area Ratio
    BI2 = area / perimeter

    # BI3: Compactness 
    BI3 = (4 * np.pi * area) / (perimeter * perimeter)

    # BI1: Fit ellipse
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)

        ellipse_mask = np.zeros_like(mask)
        cv2.ellipse(ellipse_mask, ellipse, 255, -1)

        overlap = np.logical_xor(mask, ellipse_mask).sum()
        BI1 = overlap / area
    else:
        BI1 = 0

    return BI1, BI2, BI3


def compute_diameter_features(mask):

    props = regionprops(mask.astype(np.uint8))[0]

    a = props.major_axis_length
    b = props.minor_axis_length

    # D1: Estimated diameter (average of 2 diameters)
    D1 = (a + b) / 2

    # D2: Difference between major/minor axes
    D2 = a - b

    return D1, D2

def compute_color_score(img, mask):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bin_mask = mask > 0  
    lesion_pixels = img_rgb[bin_mask]
    lesion_pixels = lesion_pixels / 255.0

    total_pixels = len(lesion_pixels)
    if total_pixels == 0:
        return 0

    R = lesion_pixels[:, 0]
    G = lesion_pixels[:, 1]
    B = lesion_pixels[:, 2]

    colours = [
        (R >= 0.7) & (G >= 0.7) & (B >= 0.7), #white
        (R >= 0.588) & (G < 0.2) & (B < 0.2), #red
        (R >= 0.588) & (R <= 0.94) & (G > 0.196) & (G <= 0.588) & (B > 0.0) & (B < 0.392), #light brown
        (R > 0.243) & (R < 0.56) & (G >= 0.0) & (G < 0.392) & (B > 0.0) & (B < 0.392), #dark brown
        (R >= 0.0) & (R <= 0.588) & (G >= 0.392) & (G <= 0.588) & (B >= 0.490) & (B <= 0.588), #blue
        (R <= 0.243) & (G <= 0.243) & (B <= 0.243) #black
    ]

    score = 0
    
    for condition in colours:
        count = np.sum(condition)
        if (count / total_pixels) >= 0.02:
            score += 1

    return score

def rotate_to_major_axis(mask):
    """
    Rotates the lesion so that:
      - The MAJOR AXIS is horizontal
      - The LONGEST DIAMETER lies on the horizontal axis
    Returns a BGR image with axes + contour drawn.
    """
    # Convert to binary boolean mask
    bin_mask = mask > 0

    # Region properties: centroid, major axis angle, etc
    labeled = label(bin_mask)
    props = regionprops(labeled)[0]

    cy, cx = props.centroid
    orientation = props.orientation   # In radians (measured CCW from horizontal)

    # orientation is negative when object is leaning right
    angle_deg = -np.degrees(orientation)

    # --- IMPORTANT ---
    # If the major axis is VERTICAL (≈ ±90°), rotate an extra 90°
    # This ensures the longest axis becomes horizontal
    angle_deg += 270 if angle_deg < 0 else 180

    # Apply rotation
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    rotated = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))

    # Convert to BGR for drawing
    vis = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)

    # Recompute centroid after rotation
    labeled_rot = label(rotated > 0)
    props_rot = regionprops(labeled_rot)[0]
    cy_r, cx_r = map(int, props_rot.centroid)

    # Draw horizontal + vertical axes
    cv2.line(vis, (0, cy_r), (vis.shape[1], cy_r), (0, 255, 0), 2)   # horizontal
    cv2.line(vis, (cx_r, 0), (cx_r, vis.shape[0]), (0, 255, 0), 2)   # vertical

    # Draw contour on rotated mask
    contours, _ = cv2.findContours((rotated > 0).astype(np.uint8),
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 0, 255), 2)

    return vis

def label_from_diagnosis(diagnosis_1):
    diagnosis = str(diagnosis_1).lower()

    malignant_terms = [
        "melanoma",
        "malignant",
        "carcinoma"
    ]

    for term in malignant_terms:
        if term in diagnosis:
            return 1 

    return 0 



"""
if __name__ == "__main__":
    csv_file = open("lesion_features.csv", mode="w", newline="")
    csv_writer = csv.writer(csv_file)

    # Write the column names
    csv_writer.writerow([
    "AS1", "AS2",
    "BI1", "BI2", "BI3",
    "D1", "D2",
    "ColorScore",
    "Label"
    ])
    
    filename = 'HAM10000\ISIC_0035995.jpg'
    img = cv2.imread(filename)
    if img is not None:
        final_img = remove_hairs(img)
        mask = contour_sauvola(final_img).astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        largest_contour = sorted_contours[0]
        
        image_countours = final_img.copy()
        cv2.drawContours(image_countours, contours, -1, (0, 255, 0), 2)
        
        # Create a mask for the largest contour
        final_mask = np.zeros_like(mask)
        
        cv2.drawContours(final_mask, [largest_contour], 0, (255), thickness=cv2.FILLED)
        
        AS1, AS2 = compute_asymmetry(final_mask)
        print(f"Asymmetry Scores: AS1 = {AS1}, AS2 = {AS2}")
        BI1, BI2, BI3 = compute_border_features(final_mask, largest_contour)
        print(f"Border Irregularity Features: BI1 = {BI1}, BI2 = {BI2}, BI3 = {BI3}")
        D1, D2 = compute_diameter_features(final_mask)
        print(f"Diameter Features: D1 = {D1}, D2 = {D2}")
        color_score = compute_color_score(img, final_mask)
        print(f"Color Score: {color_score}")
        
        
        cv2.imshow("Original", img)
        cv2.imshow("Result", final_img)
        cv2.imshow("Contour Mask", image_countours)
        cv2.imshow("Largest Contour Mask", final_mask)
        aligned_view = rotate_to_major_axis(final_mask)
        cv2.imshow("Aligned Lesion with Axes", aligned_view)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

        
        meta = pd.read_csv("ISIC_metadata.csv")

        # Create dictionary: "ISIC_0024306" → "Benign"
        diagnosis_dict = dict(zip(meta["isic_id"], meta["diagnosis_1"]))
        
        img_id = filename.replace(".jpg", "")  # e.g., "ISIC_0035995"
        diagnosis_1 = diagnosis_dict[img_id]

        label = label_from_diagnosis(diagnosis_1)
        
        csv_writer.writerow([
            AS1, AS2,
            BI1, BI2, BI3,
            D1, D2,
            color_score,
            label
        ])
"""