import math
import numpy as np
import matplotlib.pyplot as plt
import cv2

image_path = "f4.jpg"
img_bgr = cv2.imread(image_path)

if img_bgr is None:
    raise FileNotFoundError("Fail to load image.")

def manual_resize(image, new_width, new_height):
    old_height, old_width = image.shape[:2]
    resized = np.zeros((new_height, new_width, 3), dtype=image.dtype)
    scale_x = old_width / new_width
    scale_y = old_height / new_height
    for y in range(new_height):
        for x in range(new_width):
            src_x = int(x * scale_x)
            src_y = int(y * scale_y)
            resized[y, x] = image[src_y, src_x]
    return resized

def bgr_to_lab_manual(bgr_image):
    bgr = bgr_image.astype(np.float32) / 255.0
    rgb = bgr[..., ::-1]
    R = rgb[..., 0]
    G = rgb[..., 1]
    B = rgb[..., 2]
    X = 0.4124*R + 0.3576*G + 0.1805*B
    Y = 0.2126*R + 0.7152*G + 0.0722*B
    Z = 0.0193*R + 0.1192*G + 0.9505*B
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    X /= (Xn + 1e-8)
    Y /= (Yn + 1e-8)
    Z /= (Zn + 1e-8)
    def f(t):
        delta = 6/29
        if t > delta**3:
            return t**(1/3)
        else:
            return t/(3*delta**2) + 4/29
    fX = np.vectorize(f)(X)
    fY = np.vectorize(f)(Y)
    fZ = np.vectorize(f)(Z)
    L = 116*fY - 16
    a = 500*(fX - fY)
    b = 200*(fY - fZ)
    L = (L/100.0)*255.0
    a = a + 128
    b = b + 128
    lab = np.dstack([L, a, b])
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return lab

def manual_clahe_l_channel(l_channel, clip_limit=2.0, tile_grid_size=(8,8)):
    h, w = l_channel.shape
    tile_h, tile_w = tile_grid_size
    block_h = h // tile_h
    block_w = w // tile_w
    result = np.zeros_like(l_channel, dtype=np.uint8)
    for by in range(tile_h):
        for bx in range(tile_w):
            start_y = by * block_h
            start_x = bx * block_w
            end_y = (by+1)*block_h if by<tile_h-1 else h
            end_x = (bx+1)*block_w if bx<tile_w-1 else w
            block = l_channel[start_y:end_y, start_x:end_x]
            hist, _ = np.histogram(block.flatten(), bins=256, range=(0,256))
            absolute_clip_val = clip_limit * (block.size / 256.0)
            excess = 0
            for i in range(256):
                if hist[i] > absolute_clip_val:
                    excess += (hist[i] - absolute_clip_val)
                    hist[i] = absolute_clip_val
            redist = excess / 256.0
            for i in range(256):
                hist[i] += redist
            hist = np.floor(hist)
            cdf = np.cumsum(hist)
            cdf = cdf / cdf[-1]
            block_flat = block.flatten()
            eq_block_flat = np.floor(cdf[block_flat]*255).astype(np.uint8)
            eq_block = eq_block_flat.reshape(block.shape)
            result[start_y:end_y, start_x:end_x] = eq_block
    return result

def manual_gaussian_blur(image, ksize=5, sigma=1.0):
    kernel_1d = np.zeros(ksize, dtype=np.float32)
    center = ksize // 2
    sum_val = 0.0
    for i in range(ksize):
        x = i - center
        val = math.exp(-(x*x)/(2*sigma*sigma))
        kernel_1d[i] = val
        sum_val += val
    kernel_1d /= sum_val
    h, w, c = image.shape
    temp = np.zeros_like(image, dtype=np.float32)
    out = np.zeros_like(image, dtype=np.float32)
    for y in range(h):
        for x in range(w):
            for ch in range(c):
                acc = 0.0
                for k in range(ksize):
                    xx = x + (k-center)
                    xx = max(0, min(xx, w-1))
                    acc += image[y, xx, ch] * kernel_1d[k]
                temp[y, x, ch] = acc
    for x in range(w):
        for y in range(h):
            for ch in range(c):
                acc = 0.0
                for k in range(ksize):
                    yy = y + (k-center)
                    yy = max(0, min(yy, h-1))
                    acc += temp[yy, x, ch] * kernel_1d[k]
                out[y, x, ch] = acc
    return np.clip(out, 0, 255).astype(np.uint8)

def bgr_to_hsv_manual(bgr_image):
    bgr = bgr_image.astype(np.float32)
    R = bgr[..., 2]
    G = bgr[..., 1]
    B = bgr[..., 0]
    Max = np.max(bgr, axis=-1)
    Min = np.min(bgr, axis=-1)
    diff = Max - Min
    V = Max
    S = np.zeros_like(V)
    nonzero_mask = (Max != 0)
    S[nonzero_mask] = (diff[nonzero_mask]/Max[nonzero_mask])*255
    H = np.zeros_like(V)
    mask_diff = (diff != 0)
    idx_r = (Max==R) & mask_diff
    idx_g = (Max==G) & mask_diff
    idx_b = (Max==B) & mask_diff
    H[idx_r] = 60*((G[idx_r] - B[idx_r]) / diff[idx_r])
    H[idx_g] = 60*((B[idx_g] - R[idx_g]) / diff[idx_g] + 2)
    H[idx_b] = 60*((R[idx_b] - G[idx_b]) / diff[idx_b] + 4)
    H = (H/2.0)
    H[H<0] += 180
    hsv = np.dstack([
        np.clip(H, 0, 180),
        np.clip(S, 0, 255),
        np.clip(V, 0, 255)
    ]).astype(np.uint8)
    return hsv

def bgr_to_ycrcb_manual(bgr_image):
    bgr = bgr_image.astype(np.float32)
    R = bgr[...,2]
    G = bgr[...,1]
    B = bgr[...,0]
    Y  =  0.299   * R + 0.587   * G + 0.114   * B
    Cr =  0.50059 * R - 0.41853 * G - 0.08106 * B + 128.0
    Cb = -0.16894 * R - 0.33166 * G + 0.50060 * B + 128.0
    ycrcb = np.dstack([
        np.clip(Y,  0,255),
        np.clip(Cr, 0,255),
        np.clip(Cb, 0,255)
    ]).astype(np.uint8)
    return ycrcb

def in_range_manual(image, lower, upper):
    h, w, c = image.shape
    mask = np.ones((h, w), dtype=np.uint8)*255
    for ch in range(c):
        mask = mask & ((image[..., ch]>=lower[ch]) & (image[..., ch]<=upper[ch]))
    return (mask*255).astype(np.uint8)

def otsu_threshold_manual(gray_image):
    hist, _ = np.histogram(gray_image.flatten(), bins=256, range=(0,256))
    total = gray_image.size
    sum_total = np.dot(np.arange(256), hist)
    current_max = 0
    threshold = 0
    sum_fore = 0
    weight_fore = 0
    for i in range(256):
        weight_fore += hist[i]
        if weight_fore == 0:
            continue
        weight_back = total - weight_fore
        if weight_back == 0:
            break
        sum_fore += i*hist[i]
        mean_fore = sum_fore/weight_fore
        mean_back = (sum_total - sum_fore)/weight_back
        between_var = weight_fore*weight_back*((mean_fore - mean_back)**2)
        if between_var > current_max:
            current_max = between_var
            threshold = i
    _, bin_mask = cv2_threshold_like(gray_image, threshold)
    return bin_mask

def cv2_threshold_like(gray_image, thresh):
    h, w = gray_image.shape
    out = np.zeros((h,w), dtype=np.uint8)
    out[gray_image>thresh] = 255
    return thresh, out

def generate_ellipse_kernel(kh, kw):
    center_y, center_x = kh//2, kw//2
    y, x = np.ogrid[:kh, :kw]
    dist = ((y-center_y)**2 / (center_y**2+1e-9) +
            (x-center_x)**2 / (center_x**2+1e-9))
    kernel = np.zeros((kh, kw), dtype=np.uint8)
    kernel[dist <= 1.0] = 1
    return kernel

def dilation_manual(binary_mask, kernel):
    kh, kw = kernel.shape
    pad_h, pad_w = kh//2, kw//2
    h, w = binary_mask.shape
    out = np.zeros_like(binary_mask)
    for y in range(h):
        for x in range(w):
            if binary_mask[y,x]==255:
                for ky in range(kh):
                    for kx in range(kw):
                        if kernel[ky,kx]==1:
                            ny = y+(ky-pad_h)
                            nx = x+(kx-pad_w)
                            if 0<=ny<h and 0<=nx<w:
                                out[ny,nx] = 255
    return out

def erosion_manual(binary_mask, kernel):
    kh, kw = kernel.shape
    pad_h, pad_w = kh//2, kw//2
    h, w = binary_mask.shape
    out = np.ones_like(binary_mask)*255
    for y in range(h):
        for x in range(w):
            hit = True
            for ky in range(kh):
                for kx in range(kw):
                    if kernel[ky,kx]==1:
                        ny = y+(ky-pad_h)
                        nx = x+(kx-pad_w)
                        if (ny<0 or ny>=h or nx<0 or nx>=w or 
                            binary_mask[ny,nx]==0):
                            hit = False
                            break
                if not hit:
                    break
            if not hit:
                out[y,x] = 0
    return out

def morph_close_manual(binary_mask, kernel):
    dilated = dilation_manual(binary_mask, kernel)
    closed = erosion_manual(dilated, kernel)
    return closed

def region_growing_manual(mask, hsv_image, seed_point, lower_hsv, upper_hsv):
    h, w = mask.shape
    region_mask = np.zeros((h,w), dtype=np.uint8)
    queue = [seed_point]
    while queue:
        x, y = queue.pop(0)
        if x<0 or x>=w or y<0 or y>=h:
            continue
        if region_mask[y,x]==255:
            continue
        if mask[y,x]==255:
            px_h, px_s, px_v = hsv_image[y,x]
            if (px_h>=lower_hsv[0] and px_h<=upper_hsv[0] and
                px_s>=lower_hsv[1] and px_s<=upper_hsv[1] and
                px_v>=lower_hsv[2] and px_v<=upper_hsv[2]):
                region_mask[y,x] = 255
                queue.append((x+1,y))
                queue.append((x-1,y))
                queue.append((x,y+1))
                queue.append((x,y-1))
    return region_mask

def find_largest_connected_component(binary_mask):
    visited = np.zeros_like(binary_mask, dtype=bool)
    h, w = binary_mask.shape
    largest_size = 0
    largest_component = []
    for y in range(h):
        for x in range(w):
            if binary_mask[y,x] == 255 and not visited[y,x]:
                queue = [(x,y)]
                visited[y,x] = True
                comp_pixels = [(x,y)]
                while queue:
                    cx, cy = queue.pop(0)
                    for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                        nx, ny = cx+dx, cy+dy
                        if 0<=nx<w and 0<=ny<h:
                            if binary_mask[ny,nx]==255 and not visited[ny,nx]:
                                visited[ny,nx] = True
                                queue.append((nx,ny))
                                comp_pixels.append((nx,ny))
                if len(comp_pixels)>largest_size:
                    largest_size = len(comp_pixels)
                    largest_component = comp_pixels
    return largest_component

def convex_hull_manual(points):
    uniq_pts = sorted(set(points))
    if len(uniq_pts)<=1:
        return uniq_pts
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower = []
    for p in uniq_pts:
        while len(lower)>=2 and cross(lower[-2], lower[-1], p)<=0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(uniq_pts):
        while len(upper)>=2 and cross(upper[-2], upper[-1], p)<=0:
            upper.pop()
        upper.append(p)
    return lower[:-1]+upper[:-1]

def convexity_defects_manual(contour_points, hull_points):
    hull_len = len(hull_points)
    if hull_len<3:
        return []
    defects = []
    point_set = set(contour_points)
    for i in range(hull_len):
        p1 = hull_points[i]
        p2 = hull_points[(i+1)%hull_len]
        minx, maxx = min(p1[0], p2[0]), max(p1[0], p2[0])
        miny, maxy = min(p1[1], p2[1]), max(p1[1], p2[1])
        far_point = None
        far_dist = 0
        candidate_points = []
        for cp in contour_points:
            if minx<=cp[0]<=maxx and miny<=cp[1]<=maxy:
                candidate_points.append(cp)
        for cp in candidate_points:
            dist_val = point_segment_distance(cp, p1, p2)
            if dist_val>far_dist:
                far_dist = dist_val
                far_point = cp
        if far_point is not None and far_dist>0:
            a = euclidean_distance(p1, far_point)
            b = euclidean_distance(p2, far_point)
            c = euclidean_distance(p1, p2)
            if 2*a*b!=0:
                cos_val = (a*a + b*b - c*c)/(2*a*b)
                cos_val = min(1.0, max(-1.0, cos_val))
                angle = math.degrees(math.acos(cos_val))
                if angle<90:
                    defects.append((p1, p2, far_point, far_dist))
    return defects

def point_segment_distance(pt, seg_start, seg_end):
    px, py = pt
    x1, y1 = seg_start
    x2, y2 = seg_end
    seg_len2 = (x2-x1)**2 + (y2-y1)**2
    if seg_len2==0:
        return euclidean_distance(pt, seg_start)
    t = max(0, min(1, ((px - x1)*(x2 - x1)+(py - y1)*(y2 - y1))/seg_len2))
    projx = x1 + t*(x2-x1)
    projy = y1 + t*(y2-y1)
    return math.sqrt((px-projx)**2 + (py-projy)**2)

def euclidean_distance(a, b):
    return math.dist(a,b)

def skin_and_fingers_manual_pipeline(bgr_image):
    img_resized = manual_resize(bgr_image, 640, 480)
    lab_img = bgr_to_lab_manual(img_resized)
    L, A, B = lab_img[...,0], lab_img[...,1], lab_img[...,2]
    L_eq = manual_clahe_l_channel(L, clip_limit=2.0, tile_grid_size=(8,8))
    lab_eq = np.dstack([L_eq, A, B]).astype(np.uint8)
    img_eq = lab_eq.copy()
    img_blurred = manual_gaussian_blur(img_eq, ksize=5, sigma=1.0)
    hsv_img = bgr_to_hsv_manual(img_blurred)
    ycrcb_img = bgr_to_ycrcb_manual(img_blurred)
    lower_hsv = np.array([0, 10, 40], dtype=np.uint8)
    upper_hsv = np.array([25,255,255], dtype=np.uint8)
    mask_hsv = in_range_manual(hsv_img, lower_hsv, upper_hsv)
    lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
    upper_ycrcb = np.array([255,173,127], dtype=np.uint8)
    mask_ycrcb = in_range_manual(ycrcb_img, lower_ycrcb, upper_ycrcb)
    combined_mask = (mask_hsv & mask_ycrcb)
    bgr_f = img_blurred.astype(np.float32)
    gray_manual = (0.299*bgr_f[...,2] + 0.587*bgr_f[...,1] + 0.114*bgr_f[...,0])
    gray_manual = np.clip(gray_manual, 0, 255).astype(np.uint8)
    mask_otsu = otsu_threshold_manual(gray_manual)
    combined_mask = (combined_mask & mask_otsu)
    kernel = generate_ellipse_kernel(3,3)
    refined_mask = morph_close_manual(combined_mask, kernel)
    seed_point = (320,240)
    final_region_mask = region_growing_manual(refined_mask, hsv_img, seed_point, lower_hsv, upper_hsv)
    if np.count_nonzero(final_region_mask)==0:
        final_region_mask = refined_mask
    final_result = np.zeros_like(img_resized)
    final_result[final_region_mask==255] = img_resized[final_region_mask==255]
    return (img_resized,
            mask_hsv, mask_ycrcb, mask_otsu,
            refined_mask, final_region_mask,
            final_result)

def count_fingers_with_convexity_manual(region_mask):
    largest_component = find_largest_connected_component(region_mask)
    if len(largest_component)==0:
        return 0
    hull_pts = convex_hull_manual(largest_component)
    if len(hull_pts)<3:
        return 0
    defects = convexity_defects_manual(largest_component, hull_pts)
    return len(defects)

(original_resized,
 mask_hsv, mask_ycrcb, mask_otsu,
 refined_mask, region_mask,
 final_result) = skin_and_fingers_manual_pipeline(img_bgr)

num_fingers = count_fingers_with_convexity_manual(region_mask)

plt.figure(figsize=(18, 9))

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(original_resized, cv2.COLOR_BGR2RGB))
plt.title("Resized Original")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(mask_hsv, cmap="gray")
plt.title("HSV Mask (manual)")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(mask_ycrcb, cmap="gray")
plt.title("YCrCb Mask (manual)")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(mask_otsu, cmap="gray")
plt.title("Otsu Mask (manual)")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(refined_mask, cmap="gray")
plt.title("Refined Mask (manual close)")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB))
plt.title("Final Skin Region (Region Growing)")
plt.axis("off")

plt.tight_layout()
plt.show()

print("Number of fingers detected:", num_fingers)

plt.figure()
plt.imshow(region_mask, cmap="gray")
plt.title("Region Mask")
plt.axis("off")
plt.show()
