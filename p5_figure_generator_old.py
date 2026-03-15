import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

IMG_PATH = "pc5-2026-3.png"  # change if needed

# -----------------------------
# Helpers
# -----------------------------
def measure_scale_bar_pixels(gray):
    """
    Measures the pixel length of the white 'd' scale bar in the top-right of the image.
    Returns (d_px, (x_min, x_max, y_draw), mask_roi, roi_origin)
    """
    h, w = gray.shape

    # ROI tuned for this figure: top band + right side (where the scale bar is)
    y1, y2 = 0, int(0.18 * h)
    x1, x2 = int(0.45 * w), w

    roi = gray[y1:y2, x1:x2]

    # Threshold: the scale bar is bright on a dark background
    thr = 180
    mask = (roi > thr).astype(np.uint8) * 255

    # Exclude the upper part where the "d" label sits (keep the band where the bar lies)
    roi_h = roi.shape[0]
    y_band0 = int(0.25 * roi_h)
    band = mask[y_band0:, :]

    ys, xs = np.where(band > 0)
    if len(xs) == 0:
        raise RuntimeError("Could not find bright pixels for the scale bar. Try adjusting threshold/ROI.")

    x_min = x1 + int(xs.min())
    x_max = x1 + int(xs.max())
    d_px = (x_max - x_min + 1)

    y_draw = y1 + y_band0 + int(np.median(ys))

    return d_px, (x_min, x_max, y_draw), mask, (x1, y1)

def measure_big_circle_pixels(gray):
    """
    Uses HoughCircles to detect the large disk (sphere image) and returns (xc, yc, r, D_px).
    """
    h, w = gray.shape

    blur = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=200,
        param1=100,
        param2=30,
        minRadius=int(0.20 * min(h, w)),
        maxRadius=int(0.55 * min(h, w)),
    )

    if circles is None:
        raise RuntimeError("HoughCircles failed. Try adjusting parameters.")

    circles = circles[0]
    valid = []
    for x, y, r in circles:
        # keep circles fully inside the image
        if (x - r) >= 0 and (x + r) < w and (y - r) >= 0 and (y + r) < h:
            valid.append((float(x), float(y), float(r)))

    if not valid:
        raise RuntimeError("No valid circle fully inside the image.")

    # choose the largest valid circle (this should be the big disk)
    xc, yc, r = max(valid, key=lambda c: c[2])
    D_px = 2.0 * r
    return xc, yc, r, D_px

def show_overlay_measurements(img_bgr, d_info, circle_info):
    """
    Plot: original image with overlays for d and D.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape

    d_px, (x_min, x_max, y_d), _, _ = d_info
    xc, yc, r, D_px = circle_info

    plt.figure(figsize=(12, 7))
    plt.imshow(img_rgb)
    plt.axis("off")

    # Overlay for d (scale bar)
    plt.plot([x_min, x_max], [y_d, y_d], linewidth=2)
    plt.text(x_min, y_d - 25, f"d = {d_px:.1f} px", fontsize=12, color='blue')

    # Overlay for D (disk diameter) - horizontal diameter through detected center
    xL, xR = xc - r, xc + r
    plt.plot([xL, xR], [yc, yc], linewidth=2)
    plt.text(xL, yc - 25, f"D = {D_px:.1f} px", fontsize=12, color='orange')

    # Draw circle perimeter for visual confirmation
    t = np.linspace(0, 2 * np.pi, 500)
    plt.plot(xc + r * np.cos(t), yc + r * np.sin(t), linewidth=1)

    ratio = D_px / d_px
    plt.text(20, h - 40, f"D/d = {ratio:.4f}", fontsize=14)

    plt.show()

def show_zoom_scale_bar(img_bgr, d_info):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    d_px, (x_min, x_max, y_d), _, _ = d_info

    pad_x, pad_y = 80, 80
    y1 = max(0, y_d - pad_y)
    y2 = min(img_rgb.shape[0], y_d + pad_y)
    x1 = max(0, x_min - pad_x)
    x2 = min(img_rgb.shape[1], x_max + pad_x)

    crop = img_rgb[y1:y2, x1:x2]

    plt.figure(figsize=(10, 4))
    plt.imshow(crop)
    plt.title("Zoom-in: scale bar used for d (with detected endpoints)")
    plt.axis("off")

    # overlay in crop coordinates
    plt.plot([x_min - x1, x_max - x1], [y_d - y1, y_d - y1], linewidth=2)
    plt.text(5, 20, f"d = {d_px:.1f} px", fontsize=12)

    plt.show()

def show_zoom_disk(img_bgr, circle_info):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    xc, yc, r, D_px = circle_info

    pad = int(0.15 * r)
    y1 = int(max(0, yc - r - pad))
    y2 = int(min(img_rgb.shape[0], yc + r + pad))
    x1 = int(max(0, xc - r - pad))
    x2 = int(min(img_rgb.shape[1], xc + r + pad))

    crop = img_rgb[y1:y2, x1:x2]

    plt.figure(figsize=(8, 8))
    plt.imshow(crop)
    plt.title("Zoom-in: detected big disk (circle fit + diameter)")
    plt.axis("off")

    # overlay in crop coords
    t = np.linspace(0, 2 * np.pi, 500)
    plt.plot((xc - x1) + r * np.cos(t), (yc - y1) + r * np.sin(t), linewidth=1)

    plt.plot([(xc - x1) - r, (xc - x1) + r], [(yc - y1), (yc - y1)], linewidth=2)
    plt.text(10, 30, f"D = {D_px:.1f} px", fontsize=12)

    plt.show()

def sketch_system(n=1.32):
    """
    Lateral (2D) sketch of the geometry:
    sphere cross-section, source on back surface, screen plane at front tangent,
    and a few representative rays.
    """
    R = 1.0
    theta_c = math.asin(1.0 / n)

    S = np.array([-R, 0.0])    # source at back pole
    screen_x = +R             # screen plane

    # Sphere outline
    t = np.linspace(0, 2 * np.pi, 800)
    xC = R * np.cos(t)
    yC = R * np.sin(t)

    plt.figure(figsize=(8, 6))
    plt.plot(xC, yC, linewidth=2)
    plt.gca().set_aspect("equal", "box")

    # Axes and screen
    plt.axhline(0, linewidth=1)
    plt.axvline(0, linewidth=1)
    plt.axvline(screen_x, linestyle="--", linewidth=2, color='black')

    # Source
    plt.scatter([S[0]], [S[1]], s=60)
    plt.text(S[0] - 0.22, S[1] - 0.10, "source", color='blue')
    plt.text(screen_x + 0.03, 0.05, "screen", color='black')

    def draw_escape_ray(a):
        u = np.array([math.cos(a), math.sin(a)])
        P = S + 2 * R * math.cos(a) * u

        t_out = math.asin(n * math.sin(a))
        delta = 2 * a - t_out
        v = np.array([math.cos(delta), math.sin(delta)])

        s = (screen_x - P[0]) / v[0]
        Q = P + s * v

        plt.plot([S[0], P[0]], [S[1], P[1]], linewidth=1.5, color='purple')
        plt.plot([P[0], Q[0]], [P[1], Q[1]], linewidth=1.5, color='purple')

    def draw_TIR_ray(a):
        u = np.array([math.cos(a), math.sin(a)])
        P = S + 2 * R * math.cos(a) * u
        nvec = P / R

        u_ref = u - 2 * np.dot(u, nvec) * nvec

        t_hit = -2.0 * np.dot(P, u_ref)
        P2 = P + t_hit * u_ref

        plt.plot([S[0], P[0]], [S[1], P[1]], linewidth=1.5, color='purple')
        plt.plot([P[0], P2[0]], [P[1], P2[1]], linewidth=1.5, color='purple')

    draw_escape_ray(0.45 * theta_c)
    draw_escape_ray(0.85 * theta_c)
    draw_TIR_ray(1.20 * theta_c)

    draw_escape_ray(-0.45 * theta_c)
    draw_escape_ray(-0.85 * theta_c)
    draw_TIR_ray(-1.20 * theta_c)

    plt.xlim(-1.25, 1.35)
    plt.ylim(-1.05, 1.05)
    plt.title("Lateral sketch of the optical geometry (2D cross-section)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

# -----------------------------
# Run
# -----------------------------
img_bgr = cv2.imread(IMG_PATH)
if img_bgr is None:
    raise FileNotFoundError(f"Could not read image at: {IMG_PATH}")

gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

d_info = measure_scale_bar_pixels(gray)
circle_info = measure_big_circle_pixels(gray)

d_px = d_info[0]
D_px = circle_info[3]
ratio = D_px / d_px

print(f"Measured scale bar length d = {d_px:.1f} pixels")
print(f"Measured disk diameter D = {D_px:.1f} pixels")
print(f"Measured ratio D/d = {ratio:.4f}")

show_overlay_measurements(img_bgr, d_info, circle_info)
show_zoom_scale_bar(img_bgr, d_info)
show_zoom_disk(img_bgr, circle_info)

sketch_system(n=1.32)