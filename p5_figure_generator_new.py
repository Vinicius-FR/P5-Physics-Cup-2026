import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import math

IMG_PATH = "pc5-2026--1024x723.png"
SAVE_FIGURES = True
SHOW_FIGURES = True

MEASURE_FIG_NAME = "p5_measure.png"
SKETCH_FIG_NAME = "p5_lateral_sketch.png"


def measure_scale_bar_pixels(gray):
    h, w = gray.shape

    x0 = int(0.35 * w)
    y0 = 0
    y1 = int(0.16 * h)
    roi = gray[y0:y1, x0:w]

    mask = (roi > 150).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    best = None
    for i in range(1, num_labels):
        x, y, ww, hh, area = stats[i]
        if area > 200 and ww > 5 * hh:
            if best is None or area > best[-1]:
                best = (x, y, ww, hh, area)

    if best is None:
        raise RuntimeError("Could not detect the scale bar.")

    x, y, ww, hh, area = best
    x_min = x0 + x
    x_max = x0 + x + ww - 1
    y_draw = y0 + y + hh // 2
    d_px = float(ww)

    return d_px, (x_min, x_max, y_draw)


def detect_circle(gray, rmin, rmax, dp=1.2, minDist=100, param1=100, param2=30):
    blur = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=minDist,
        param1=param1,
        param2=param2,
        minRadius=rmin,
        maxRadius=rmax,
    )

    if circles is None:
        raise RuntimeError("HoughCircles failed.")

    circles = circles[0]
    idx = np.argmax(circles[:, 2])
    x, y, r = circles[idx]
    return float(x), float(y), float(r)


def transmitted_screen_y_over_R(a, n, lam):
    t = np.arcsin(np.clip(n * np.sin(a), -1.0, 1.0))
    delta = 2.0 * a - t
    return np.sin(2.0 * a) + (lam + 1.0 - np.cos(2.0 * a)) * np.tan(delta)


def radii_from_n_lam(n, lam, num=200000):
    theta_c = np.arcsin(1.0 / n)
    a = np.linspace(0.0, theta_c, num)
    y = transmitted_screen_y_over_R(a, n, lam)

    i_c = np.argmax(y)
    i_o = np.argmin(y)

    rho_c = float(y[i_c])      # caustic radius / R
    rho_o = float(-y[i_o])     # outer radius / R
    a_c = float(a[i_c])

    return theta_c, a, y, a_c, rho_c, rho_o


def solve_n_and_lam(target_rho_c_over_R, target_rho_o_over_R):
    def residuals(vars_):
        n, lam = vars_
        theta_c, a, y, a_c, rho_c, rho_o = radii_from_n_lam(n, lam, num=60000)
        return np.array([
            rho_c - target_rho_c_over_R,
            rho_o - target_rho_o_over_R
        ])

    sol = least_squares(
        residuals,
        x0=np.array([2.1, 2.7]),
        bounds=([1.0001, 0.0], [10.0, 20.0]),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )

    if not sol.success:
        raise RuntimeError("Parameter fit did not converge.")

    n, lam = sol.x
    return float(n), float(lam)


def ray_data(a, n):
    R = 1.0
    S = np.array([-R, 0.0])
    u = np.array([np.cos(a), np.sin(a)])

    P = S + 2.0 * R * np.cos(a) * u

    t_out = np.arcsin(np.clip(n * np.sin(a), -1.0, 1.0))
    delta = 2.0 * a - t_out
    v = np.array([np.cos(delta), np.sin(delta)])

    return S, P, v


def finish_figure(fig, filename):
    if SAVE_FIGURES:
        fig.savefig(filename, dpi=200, bbox_inches="tight")
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


def make_measure_figure(img_bgr, d_info, outer_circle, inner_circle, values, outname):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    d_px, (x_min, x_max, y_bar) = d_info
    xo, yo, ro = outer_circle
    xi, yi, ri = inner_circle

    Dc_px = 2.0 * ri
    Do_px = 2.0 * ro

    # fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig, axes = plt.subplots(1, 1, figsize=(12, 7))

    # main overlay
    # ax = axes[0, 0]
    ax = axes
    ax.imshow(img_rgb)
    ax.set_title("Original figure with the measured diameters")
    ax.axis("off")

    t = np.linspace(0.0, 2.0 * np.pi, 600)

    ax.plot([x_min, x_max], [y_bar, y_bar], linewidth=2)
    ax.text(x_min, y_bar - 18, f"d = {d_px:.1f} px", fontsize=11, color='blue')

    ax.plot(xo + ro * np.cos(t), yo + ro * np.sin(t), linewidth=1.5, color='purple')
    ax.plot([xo - ro, xo + ro], [yo, yo], linewidth=2, color='purple')
    # ax.text(xo - ro, yo - 18, f"D_o = {Do_px:.1f} px", fontsize=11)

    ax.plot(xi + ri * np.cos(t), yi + ri * np.sin(t), linewidth=1.5, color='blue')
    ax.plot([xi - ri, xi + ri], [yi, yi], linewidth=2, color='blue')
    # ax.text(xi - ri, yi - 18, f"D_c = {Dc_px:.1f} px", fontsize=11)

    ax.text(
        18,
        img_rgb.shape[0] - 85,
        "\n".join([
            f"D_c / d = {values['ratio_c']:.4f}",
            f"D_o / d = {values['ratio_o']:.4f}",
        ]),
        fontsize=11,
        va="top",
        color='white'
    )

    # zoom - scale bar
    # ax = axes[0, 1]
    # pad_x, pad_y = 60, 45
    # x1 = max(0, x_min - pad_x)
    # x2 = min(img_rgb.shape[1], x_max + pad_x)
    # y1 = max(0, y_bar - pad_y)
    # y2 = min(img_rgb.shape[0], y_bar + pad_y)
    # crop = img_rgb[y1:y2, x1:x2]
    # ax.imshow(crop)
    # ax.set_title("Zoom - scale bar")
    # ax.axis("off")
    # ax.plot([x_min - x1, x_max - x1], [y_bar - y1, y_bar - y1], linewidth=2)
    # ax.text(8, 18, f"d = {d_px:.1f} px", fontsize=11)

    # zoom - inner bright circle
    # ax = axes[1, 0]
    # pad = int(0.65 * ri)
    # x1 = int(max(0, xi - ri - pad))
    # x2 = int(min(img_rgb.shape[1], xi + ri + pad))
    # y1 = int(max(0, yi - ri - pad))
    # y2 = int(min(img_rgb.shape[0], yi + ri + pad))
    # crop = img_rgb[y1:y2, x1:x2]
    # ax.imshow(crop)
    # ax.set_title("Zoom - bright inner circle")
    # ax.axis("off")
    # ax.plot((xi - x1) + ri * np.cos(t), (yi - y1) + ri * np.sin(t), linewidth=1.5)
    # ax.plot([xi - x1 - ri, xi - x1 + ri], [yi - y1, yi - y1], linewidth=2)
    # ax.text(8, 18, f"D_c = {Dc_px:.1f} px", fontsize=11)

    # zoom - outer circle
    # ax = axes[1, 1]
    # pad = int(0.10 * ro)
    # x1 = int(max(0, xo - ro - pad))
    # x2 = int(min(img_rgb.shape[1], xo + ro + pad))
    # y1 = int(max(0, yo - ro - pad))
    # y2 = int(min(img_rgb.shape[0], yo + ro + pad))
    # crop = img_rgb[y1:y2, x1:x2]
    # ax.imshow(crop)
    # ax.set_title("Zoom - outer circle")
    # ax.axis("off")
    # ax.plot((xo - x1) + ro * np.cos(t), (yo - y1) + ro * np.sin(t), linewidth=1.2)
    # ax.plot([xo - x1 - ro, xo - x1 + ro], [yo - y1, yo - y1], linewidth=2)
    # ax.text(8, 18, f"D_o = {Do_px:.1f} px", fontsize=11)

    plt.tight_layout()
    finish_figure(fig, outname)


def make_sketch_figure(n, lam, outname):
    theta_c, a_grid, y_grid, a_c, rho_c, rho_o = radii_from_n_lam(n, lam, num=200000)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))

    # left panel - geometry
    ax = axes[0]
    R = 1.0
    L = lam * R
    screen_x = R + L

    t = np.linspace(0.0, 2.0 * np.pi, 800)
    ax.plot(R * np.cos(t), R * np.sin(t), linewidth=2)

    ax.axhline(0.0, linewidth=1)
    ax.axvline(0.0, linewidth=1)
    ax.plot([screen_x, screen_x], [-1.65, 1.65], linestyle="--", linewidth=2, color='black')

    S = np.array([-R, 0.0])
    ax.scatter([S[0]], [S[1]], s=50)
    ax.text(S[0] - 0.20, S[1] - 0.20, "source")
    ax.text(screen_x + 0.01, 1.40, "screen")

    # axial ray
    S0, P0, v0 = ray_data(0.0, n)
    s0 = (screen_x - P0[0]) / v0[0]
    Q0 = P0 + s0 * v0
    ax.plot([S0[0], P0[0]], [S0[1], P0[1]], linewidth=1.5, color='purple')
    ax.plot([P0[0], Q0[0]], [P0[1], Q0[1]], linewidth=1.5, color='purple')

    # caustic rays
    for sign in [+1.0, -1.0]:
        aa = a_c
        S1, P1, v1 = ray_data(aa, n)
        P1 = P1.copy()
        v1 = v1.copy()
        P1[1] *= sign
        v1[1] *= sign
        s1 = (screen_x - P1[0]) / v1[0]
        Q1 = P1 + s1 * v1
        ax.plot([S[0], P1[0]], [S[1], P1[1]], linewidth=1.8, color='purple')
        ax.plot([P1[0], Q1[0]], [P1[1], Q1[1]], linewidth=1.8, color='purple')

    # critical rays
    for sign in [+1.0, -1.0]:
        aa = 0.999 * theta_c
        S2, P2, v2 = ray_data(aa, n)
        P2 = P2.copy()
        v2 = v2.copy()
        P2[1] *= sign
        v2[1] *= sign
        s2 = (screen_x - P2[0]) / v2[0]
        Q2 = P2 + s2 * v2
        ax.plot([S[0], P2[0]], [S[1], P2[1]], linewidth=1.6, color='blue')
        ax.plot([P2[0], Q2[0]], [P2[1], Q2[1]], linewidth=1.6, color='blue')

    # one trapped ray
    aT = 1.12 * theta_c
    u = np.array([np.cos(aT), np.sin(aT)])
    P = S + 2.0 * np.cos(aT) * u
    nvec = P / np.linalg.norm(P)
    u_ref = u - 2.0 * np.dot(u, nvec) * nvec
    t_hit = -2.0 * np.dot(P, u_ref)
    P2 = P + t_hit * u_ref
    ax.plot([S[0], P[0]], [S[1], P[1]], linewidth=1.4, linestyle=":", color='black')
    ax.plot([P[0], P2[0]], [P[1], P2[1]], linewidth=1.4, linestyle=":", color='black')
    ax.text(0.05, 0.90, "trapped ray", fontsize=10)

    # marks on the screen
    ax.plot([screen_x, screen_x], [0.0, rho_c], linewidth=1, color='purple')
    ax.plot([screen_x, screen_x], [0.0, -rho_o], linewidth=1, color='blue')
    ax.text(screen_x + 0.05, 0.50 * rho_c, r"$\rho_c$")
    ax.text(screen_x + 0.05, -0.55 * rho_o, r"$\rho_o$")

    # distance L
    ax.annotate(
        "",
        xy=(R, -1.28),
        xytext=(screen_x, -1.28),
        arrowprops=dict(arrowstyle="<->", linewidth=1.2),
    )
    ax.text(0.5 * (R + screen_x) - 0.03, -1.22, r"$L$")

    ax.set_aspect("equal", "box")
    ax.set_xlim(-1.25, screen_x + 0.45)
    ax.set_ylim(-1.55, 1.55)
    ax.set_xlabel(r"$x/R$")
    ax.set_ylabel(r"$y/R$")
    ax.set_title("Lateral sketch of the optical geometry (2D cross-section)")

    # right panel - y(a)/R
    ax = axes[1]
    ax.plot(a_grid / theta_c, y_grid, linewidth=2)
    ax.axhline(0.0, linewidth=1)
    ax.axvline(a_c / theta_c, linestyle="--", linewidth=1)
    ax.axvline(1.0, linestyle="--", linewidth=1)

    ax.plot([a_c / theta_c], [rho_c], "o", color='purple')
    ax.plot([1.0], [-rho_o], "o", color='blue')

    ax.text(a_c / theta_c + 0.02, rho_c + 0.03, r"$\rho_c/R$")
    ax.text(0.90, -rho_o, r"$-\rho_o/R$")

    ax.set_xlim(0.0, 1.02)
    ax.set_xlabel(r"$a/\theta_c$")
    ax.set_ylabel(r"$y(a)/R$")
    ax.set_title("Image positon on the screen as function of the refraction angle")

    plt.tight_layout()
    finish_figure(fig, outname)


# -----------------------------
# Run
# -----------------------------
img_bgr = cv2.imread(IMG_PATH)
if img_bgr is None:
    raise FileNotFoundError(f"Could not read image at: {IMG_PATH}")

gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# Measurements
d_info = measure_scale_bar_pixels(gray)
d_px = d_info[0]

xo, yo, ro = detect_circle(gray, rmin=220, rmax=360, param2=30, minDist=200)
xi, yi, ri = detect_circle(gray, rmin=40, rmax=100, param2=20, minDist=80)

Dc_px = 2.0 * ri
Do_px = 2.0 * ro

ratio_c = Dc_px / d_px
ratio_o = Do_px / d_px

# Optical parameters
n, lam = solve_n_and_lam(ratio_c, ratio_o)
f_sc = math.sqrt(1.0 - 1.0 / n**2)

# Prints
print(f"d   = {d_px:.1f} px")
print(f"D_c = {Dc_px:.1f} px")
print(f"D_o = {Do_px:.1f} px")
print(f"D_c / d = {ratio_c:.4f}")
print(f"D_o / d = {ratio_o:.4f}")
print(f"n = {n:.4f}")
print(f"L / R = {lam:.4f}")
print(f"f_sc = {f_sc:.4f}")

values = {
    "ratio_c": ratio_c,
    "ratio_o": ratio_o,
    "n": n,
    "lam": lam,
    "f_sc": f_sc,
}

make_measure_figure(
    img_bgr=img_bgr,
    d_info=d_info,
    outer_circle=(xo, yo, ro),
    inner_circle=(xi, yi, ri),
    values=values,
    outname=MEASURE_FIG_NAME,
)

make_sketch_figure(
    n=n,
    lam=lam,
    outname=SKETCH_FIG_NAME,
)