import cv2
import numpy as np
from skimage.segmentation import chan_vese
from skimage import exposure
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from skimage.filters import median
from skimage.color import rgb2gray
from skimage import img_as_float

def remove_hairs(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    result = cv2.inpaint(img, thresh, 3, cv2.INPAINT_TELEA)
    return result

def contour(cleaned_img):

    gray = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2GRAY)
    #gray_blur = cv2.GaussianBlur(gray, (7, 7), 0)
    #gray_eq = exposure.equalize_adapthist(gray, clip_limit=0.03)
    img_float = gray.astype(np.float32)

    h, w = img_float.shape
    Y, X = np.ogrid[:h, :w]
    center_x, center_y = w // 2, h // 2
    radius = min(h, w) // 4
    init_ls = (X - center_x)**2 + (Y - center_y)**2 < radius**2

    # Apply Chan-Vese
    img_countour = chan_vese(img_float, mu=0.1, lambda1=1, lambda2=1, 
                            tol=1e-4, max_num_iter=300, init_level_set=init_ls, extended_output=False)
    
    final_mask = remove_small_objects(label(img_countour), min_size=500)

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
    """
    img  : BGR uint8 image
    mask : binary mask (0 or 255) of the lesion
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to boolean: True where pixel belongs to mask
    bin_mask = mask > 0  

    #Extract lesion pixels from BGR → RGB
    lesion_pixels = img[bin_mask]           # shape (N, 3), still BGR
    #lesion_pixels = lesion_pixels[:, ::-1]  # convert BGR → RGB
    lesion_pixels = lesion_pixels / 255.0   # normalize to [0,1]
    print(lesion_pixels.size)

    #Color boundaries from the paper
    colors = [
        ("White",      [0.8,   0.8,   0.8],   [1, 1, 1]),
        ("Red",        [0.588, 0,     0],     [1, 0.2, 0.2]),
        ("LightBrown", [0.588, 0.196, 0],     [0.94, 0.588, 0.392]),
        ("DarkBrown",  [0.243, 0,     0],     [0.56, 0.392, 0.392]),
        ("BlueGray",   [0,     0.392, 0.490], [0.588, 0.588, 0.588]),
        ("Black",      [0,     0,     0],     [0.243, 0.243, 0.243])
    ]

    total = len(lesion_pixels)
    print(total)
    if total == 0:
        return 0

    score = 0
    for name, lo, hi in colors:
        count = 0
        lo = np.array(lo)
        hi = np.array(hi)
        for px in lesion_pixels:
            if np.all(px >= lo) and np.all(px <= hi):
                count += 1

        if ((count/total) >= 0.05):
            score += 1

    return score
    



if __name__ == "__main__":
    img = cv2.imread('HAM10000\ISIC_0033580.jpg')
    if img is not None:
        final_img = remove_hairs(img)
        mask = contour(final_img).astype(np.uint8) * 255
        
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
        cv2.waitKey(0)
        cv2.destroyAllWindows()