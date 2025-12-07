import cv2
import pandas as pd
import numpy as np
import csv
from feature_extractor import (
    remove_hairs,
    contour_sauvola,
    compute_asymmetry,
    compute_border_features,
    compute_diameter_features,
    compute_color_score,
    rotate_to_major_axis,
    label_from_diagnosis
)

#Load metadata
meta = pd.read_csv("collection_212_metadata.csv")
diagnosis_dict = dict(zip(meta["isic_id"], meta["diagnosis_1"]))


csv_file = open("lesion_features.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "AS1", "AS2",
    "BI1", "BI2", "BI3",
    "D1", "D2",
    "ColorScore",
    "Label"
])

# Choose image to process
filename = "ISIC_0024444.jpg"
img_path = f"HAM10000/{filename}"

img = cv2.imread(img_path)

if img is None:
    raise ValueError("Image not found!")


final_img = remove_hairs(img)
mask = contour_sauvola(final_img).astype(np.uint8) * 255

# Find lesion contour
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

final_mask = np.zeros_like(mask)
cv2.drawContours(final_mask, [largest_contour], 0, (255), thickness=cv2.FILLED)

# Calculate features
AS1, AS2 = compute_asymmetry(final_mask)
BI1, BI2, BI3 = compute_border_features(final_mask, largest_contour)
D1, D2 = compute_diameter_features(final_mask)
color_score = compute_color_score(img, final_mask)

img_id = filename.replace(".jpg", "")
diagnosis_1 = diagnosis_dict[img_id]
label = label_from_diagnosis(diagnosis_1)

csv_writer.writerow([
    AS1, AS2,
    BI1, BI2, BI3,
    D1, D2,
    color_score,
    label
])

csv_file.close()

print("Feature extraction complete!")
