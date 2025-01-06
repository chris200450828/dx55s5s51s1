import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

image_path = "f4.jpg"
img = cv2.imread(image_path)

if img is not None:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis("off")
    plt.show()



def combined_skin_detection_with_region_growing(image):
    img_resized = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)

    lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    img_blurred = cv2.GaussianBlur(img_eq, (5, 5), 0)
    hsv_img = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([0, 10, 40], dtype=np.uint8) 
    upper_hsv = np.array([25, 255, 255], dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv_img, lower_hsv, upper_hsv)
    ycrcb_img = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2YCrCb)
    lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)  
    upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
    mask_ycrcb = cv2.inRange(ycrcb_img, lower_ycrcb, upper_ycrcb)
    combined_mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
    gray_img = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2GRAY)
    _, mask_otsu = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTS
    combined_mask = cv2.bitwise_and(combined_mask, mask_otsu)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    refined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    def region_growing(mask, hsv_img, seed_point, lower_hsv, upper_hsv):
        region_mask = np.zeros(mask.shape, dtype=np.uint8)
        region_queue = [seed_point]

        while region_queue:
            x, y = region_queue.pop(0)

            if region_mask[y, x] == 255:
                continue

            pixel_hsv = hsv_img[y, x]
            if np.all(pixel_hsv >= lower_hsv) and np.all(pixel_hsv <= upper_hsv) and mask[y, x] == 255:
                region_mask[y, x] = 255

                if x > 0:
                    region_queue.append((x - 1, y))
                if x < mask.shape[1] - 1:
                    region_queue.append((x + 1, y))
                if y > 0:
                    region_queue.append((x, y - 1))
                if y < mask.shape[0] - 1:
                    region_queue.append((x, y + 1))

        return region_mask

    seed_point = (320, 240)
    final_region_mask = region_growing(refined_mask, hsv_img, seed_point, lower_hsv, upper_hsv)

    if np.count_nonzero(final_region_mask) == 0:
        final_region_mask = refined_mask  
    final_result = cv2.bitwise_and(img_resized, img_resized, mask=final_region_mask)

    return img_resized, mask_hsv, mask_ycrcb, mask_otsu, refined_mask, final_region_mask, final_result


original_resized, mask_hsv, mask_ycrcb, mask_otsu, refined_mask, region_mask, final_result = combined_skin_detection_with_region_growing(
    img)

plt.figure(figsize=(20, 10))
plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(original_resized, cv2.COLOR_BGR2RGB))
plt.title("Resized Original")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(mask_hsv, cmap="gray")
plt.title("HSV Mask")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(mask_ycrcb, cmap="gray")
plt.title("YCrCb Mask")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(mask_otsu, cmap="gray")
plt.title("Otsu Threshold Mask")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(refined_mask, cmap="gray")
plt.title("Refined Combined Mask")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB))
plt.title("Final Detected Skin (Region Growing)")
plt.axis("off")

plt.show()


def generate_angle_distribution(region_mask):
    """
    Generate an angle distribution based on the region mask.

    :param region_mask: Binary mask of the detected region (from region growing).
    :return: Angles (in degrees) of all detected skin pixels relative to the center.
    """
    y_coords, x_coords = np.where(region_mask == 255)
    center_x = int(np.mean(x_coords))
    center_y = int(np.mean(y_coords))

    angles = []
    for x, y in zip(x_coords, y_coords):
        angle = math.degrees(math.atan2(y - center_y, x - center_x))
        angles.append(angle)

    return angles, (center_x, center_y)


def count_fingers_with_convexity(region_mask):
    contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0

    largest_contour = max(contours, key=cv2.contourArea)

    hull = cv2.convexHull(largest_contour, returnPoints=False)

    if len(hull) > 3:
        defects = cv2.convexityDefects(largest_contour, hull)
    else:
        defects = None

    if defects is None:
        return 0

    finger_count = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(largest_contour[s][0])
        end = tuple(largest_contour[e][0])
        far = tuple(largest_contour[f][0]
        a = np.linalg.norm(np.array(start) - np.array(far))
        b = np.linalg.norm(np.array(end) - np.array(far))
        c = np.linalg.norm(np.array(start) - np.array(end))
        angle = math.degrees(math.acos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b)))

        if angle < 90:
            finger_count += 1

    return finger_count


num_fingers = count_fingers_with_convexity(region_mask)
print(f"Number of fingers detected: {num_fingers}")

plt.figure(figsize=(8, 8))
plt.imshow(region_mask, cmap="gray")
plt.title("Region Mask with Convexity Defects")
plt.axis("off")
plt.show()
