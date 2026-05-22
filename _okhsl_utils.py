from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
import sys
import matplotlib.pyplot as plt
from scipy.linalg import norm as scipy_norm


# === Internal functions  of OKhsl ===
# These functions are called by the main function emb_to_okhsl() within clrmappy

def _fit_and_center_2d3d_pca(emb_3d, emb_2d, center_around, pc1_and_2_from_2d, equal_variance_mode):
    """
    This function fits the original embedding by perfoming a pca on it an centering the pca.
    The equal variation rotation described in the main function is applied here.

    Parameters
    ----------
    emb_3d, emb_2d : array
        original embeddings
    center_around : string
        either 'mid' or 'mean'
    pc1_and_2_from_2d : bool
        decides if pc1 and 2 of the fitted embedding are computed from the 2D embedding
    euqal_variance_mode : bool
        if True applies rotation matrix, that aims to provide similar variances on each axis.

    Returns
    -----
    emb3d_fit or emb2d_fit : array; fitted embedding for further downstream fitting and analysis
    """
    # Initialise so the 3D branch below can safely test
    # `emb2d_fit is not None` for the `pc1_and_2_from_2d` case.
    emb2d_fit = None
    if emb_2d is not None:
        # 2D PCA on 2D umap
        pca_2d = PCA(n_components=2)
        emb2d_fit = pca_2d.fit_transform(emb_2d)

        if emb_3d is None:
            if center_around == 'mean':
                # centering around the mean (for each collumn respectively), subtracts the mean
                emb2d_fit -= emb2d_fit.mean(axis=0)

            elif center_around == 'mid':
                # centering around the mid of the span of the axis (for each collumn respectively), subtracts the mid
                emb2d_fit -= (emb2d_fit.max(axis=0) +
                              emb2d_fit.min(axis=0)) / 2

            return emb2d_fit

    if emb_3d is not None:
        # 3D PCA on 3D umap
        pca_3d = PCA(n_components=3)
        emb3d_fit = pca_3d.fit_transform(emb_3d)

        # If the equal variance mode is true, the rotation is applied. Only if it is false, using the pc1 and 2 from 2d is possible!
        # Because this latter mode requires the original pcs and aims at providing extra depth through using pc3 from 3d for the brightness
        if equal_variance_mode == True:

            # Rotation Matrix so that every axis involves every component (45° turn around all axes)!
            rotation_matrix = np.array([
                [0.500,  0.500, -0.707],
                [-0.146,  0.854,  0.500],
                [0.854, -0.146,  0.500],
            ])

            emb3d_fit = emb3d_fit @ rotation_matrix

            variance_per_axis = np.var(emb3d_fit, axis=0)
            total_variance = np.sum(variance_per_axis)

            print(
                "The rotation for similar variance per axis, has been applied, these are the variances:")
            print(
                f"X-Axis: {variance_per_axis[0]:.4f} ({variance_per_axis[0]/total_variance*100:.1f}%)")
            print(
                f"Y-Axis: {variance_per_axis[1]:.4f} ({variance_per_axis[1]/total_variance*100:.1f}%)")
            print(
                f"Z-Axis: {variance_per_axis[2]:.4f} ({variance_per_axis[2]/total_variance*100:.1f}%)")

        elif emb2d_fit is not None and pc1_and_2_from_2d == True:
            emb3d_fit[:, 0] = emb2d_fit[:, 0].copy()
            emb3d_fit[:, 1] = emb2d_fit[:, 1].copy()

        if center_around == 'mean':
            # centering around the mean (for each collumn respectively), subtracts the mean
            emb3d_fit -= emb3d_fit.mean(axis=0)

        elif center_around == 'mid':
            # centering around the mid of the span of the axis (for each collumn respectively), subtracts the mid
            emb3d_fit -= (emb3d_fit.max(axis=0) + emb3d_fit.min(axis=0)) / 2

        return emb3d_fit


def _find_optimal_rotation(emb3d_fit, brightness_range):
    """
    This function inhabits the rotation algorithm, that finds the optimal rotation angle for a maximum mean saturation.

    Parameters
    ----------
    emb3d_fit : array
        previously fitted 3D embedding
    brightness_range : list
        range of the brightness

    Returns
    -----
    optimal_theta: float; optimal rotation angle
    """
    z_scaled_min = brightness_range[0]
    z_scaled_max = brightness_range[1]

    # Rotate PCA around PC2
    rotation = []
    x = emb3d_fit[:, 0].copy()
    y = emb3d_fit[:, 1].copy()
    z = emb3d_fit[:, 2].copy()
    # calculates initial xz angles to the y axis of each point
    angle = np.degrees(np.arctan2(z, x))
    # calculates the distance of each point to the y axis and saves it in an array. This distance does not change with rotation around y (PC2)!
    r_y = np.sqrt(x**2 + z**2)
    # calculating the desired z delta in the final data
    z_scaled_delta = z_scaled_max - z_scaled_min

    step = 1
    for theta in np.arange(step, 180 + step, step):
        angle = angle + step
        x = r_y * np.cos(np.radians(angle))  # new x-values
        # y-value always stays the same as in the beginning!
        y = emb3d_fit[:, 1].copy()
        z = r_y * np.sin(np.radians(angle))  # new z-values

        r_z = np.sqrt(x**2 + y**2)  # distance from z-axis after rotation
        r_max = r_z.max()  # maximum distance form z-axis after rotation

        # scale according to r_max, so that the maximum distance from z-axis is 1
        x, y, z = x/r_max, y/r_max, z/r_max
        # calculate z_delta
        z_delta = z.max() - z.min()

        # scale axes if z_delta is greater than z_scaled_delta because we later want to scale z lightness to z_scaled_min - z_scaled_max
        if z_delta > z_scaled_delta:
            z = (z - z.min()) / z_delta * z_scaled_delta + z_scaled_min
            x = x / z_delta * z_scaled_delta
            y = x / z_delta * z_scaled_delta

        # recalculate the distance to the z-axis!
        r_z = np.sqrt(x**2 + y**2)
        r_mean = r_z.mean()

        rotation.append([theta, r_mean])

    # converts rotation into a numpy array
    rotation = np.array(rotation)
    angles = rotation[:, 0]
    r_mean = rotation[:, 1]

    # ── Optimal angle: find the angle with the maximum r_mean ───────────────────────────────────────

    idx = np.argmax(np.abs(r_mean))
    optimal_theta = angles[idx]
    optimal_r_mean = r_mean[idx]

    print(f"Optimal rotation: {optimal_theta:.2f}°")
    print(f"r_mean at optimal: {optimal_r_mean:.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────────────
    theta_fine = np.linspace(0, 180, 2000)
    theta_tan_rng = np.linspace(0, 90, 500)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(angles, r_mean, s=4, color='steelblue',
               label=f'data ({step}° steps)', zorder=3)
    ax.axvline(optimal_theta, color='green', linestyle='--',
               linewidth=1, label=f'optimal {optimal_theta:.2f}°')
    ax.set_xlabel('Rotation angle θ [°]')
    ax.set_ylabel('r_mean')
    ax.set_title('r_mean vs. rotation angle around PC2')
    ax.legend()
    ax.set_xlim(0, 180)
    ax.set_xticks(range(0, 180, 45))
    plt.tight_layout()
    plt.show()

    return optimal_theta


# def color_theme(embedding_3d_fit):
    x = embedding_3d_fit[:, 0].copy()
    y = embedding_3d_fit[:, 1].copy()

    angle_xy = np.degrees(np.arctan2(y, x))
    r_z = np.sqrt(x**2 + y**2)

    # rotate around z
    x = r_z * np.cos(np.radians(angle_xy))  # new x-values
    y = r_z * np.sin(np.radians(angle_xy))  # new y-values


def _apply_rotation_scaling(emb3d_fit, brightness_range):
    """
    This function applies the calculated optimal angle of the rotation algorithm to the embedding.

    Parameters
    ----------
    emb3d_fit : array
        previously fitted 3D embedding
    brightness_range : list
        range of the brightness

    Returns
    -----
    emb3d_fit : array; fitted embedding with applied rotation angle
    r_z : array; distances to the center (saturation) of each point
    """
    z_scaled_min = brightness_range[0]
    z_scaled_max = brightness_range[1]

    optimal_theta = _find_optimal_rotation(
        emb3d_fit=emb3d_fit, brightness_range=brightness_range)

    x = emb3d_fit[:, 0].copy()
    y = emb3d_fit[:, 1].copy()
    z = emb3d_fit[:, 2].copy()

    # Rotate the PCA with the calculated angle
    # calculates final xz angles to the y axis(PC2) in fit embedding
    angle = np.degrees(np.arctan2(z, x)) + optimal_theta

    # calculate the distance of each point to the y axis and save it in an array. This distance does not change with rotation around y (PC2)!
    r_y = np.sqrt(x**2 + z**2)

    # calculating the desired z delta in the final data
    z_scaled_delta = z_scaled_max - z_scaled_min

    x = r_y * np.cos(np.radians(angle))  # new x-values
    z = r_y * np.sin(np.radians(angle))  # new z-values

    r_z = np.sqrt(x**2 + y**2)  # distance from z-axis after rotation
    r_max = r_z.max()

    # scale according to r_max (so r_max becomes 1)
    x, y, z = x / r_max, y / r_max, z / r_max

    # calculate z_delta
    z_delta = z.max() - z.min()

    # scale axes if z_delta is greater than z_scaled_delta because we later want to scale z lightness to z_scaled_min - z_scaled_max
    if z_delta > z_scaled_delta:
        z = (z - z.min()) / z_delta * z_scaled_delta + z_scaled_min
        x = x / z_delta * z_scaled_delta
        y = y / z_delta * z_scaled_delta
    else:
        # centers the data, which also happens in the if case
        z = z - z.min() + z_scaled_min

    # recalculate the distance to the z-axis!
    r_z = np.sqrt(x**2 + y**2)

    print(
        f'The mean saturation before saturation enhancement  is {r_z.mean()}.')

    # save scaling in embedding_3d_fit
    emb3d_fit[:, 0] = x.copy()
    emb3d_fit[:, 1] = y.copy()
    emb3d_fit[:, 2] = z.copy()

    return {
        'embedding': emb3d_fit,
        'r_z': r_z,
    }


def _saturation_enhancement(emb_fit, saturation_range):
    """
    This function enhances the saturation by min-max-scaling.
    Parameters
    ----------
    emb3d_fit : array
        previously fitted 3D embedding
    saturation_range : list
        range of the saturation

    Returns
    -----
    emb3d_fit : array; fitted embedding 
    r_z : array; distances to the center (saturation) of each point
    """

    x = emb_fit[:, 0].copy()
    y = emb_fit[:, 1].copy()

    # calculate the distance to the z-axis!
    r_z = np.sqrt(x**2 + y**2)

    # The saturation is increased here, which is of course a slight distortion, but could benefit the visualization
    if r_z.min() < saturation_range[0]:
        sat_min = saturation_range[0]
    else:
        sat_min = r_z.min()
    sat_max = saturation_range[1]
    sat_delta = sat_max - sat_min
    r_z = (r_z - r_z.min())/(r_z.max()-r_z.min()) * \
        sat_delta + sat_min  # min max scaling

    # implement scaling in x and y
    # calculate xy angle in order to implement the scaling in x and y
    angle_xy = np.degrees(np.arctan2(y, x))

    x = r_z * np.cos(np.radians(angle_xy))  # new x-values
    y = r_z * np.sin(np.radians(angle_xy))  # new y-values

    print(f'The mean saturation after saturation enhancement is {r_z.mean()}.')

    # save scaling in embedding_3d_fit
    emb_fit[:, 0] = x.copy()
    emb_fit[:, 1] = y.copy()

    return {
        'embedding': emb_fit,
        'r_z': r_z,
    }


def _emb_to_OKhsl(emb_fit, r_z, brightness_range):
    """
    This function calculates the OKhsl coloring based on the final fitted embedding.
    Parameters
    ----------
    emb3d_fit : array
        previously fitted 3D embedding
    brightness_range : list
        range of the brightness
    r_z : array
        distances to the center (saturation) of each point

    Returns
    -----
    OKhsl : array; this is the coloring for each point, in OKhsl coordinates.
    """
    OKhsl = emb_fit.copy()

    x = emb_fit[:, 0].copy()
    y = emb_fit[:, 1].copy()

    # calculate and save hue as xy angle
    OKhsl[:, 0] = np.degrees(np.arctan2(y, x)).copy()

    # Save saturation
    OKhsl[:, 1] = r_z.copy()

    # save brightness
    if OKhsl.shape[1] == 2:
        mid_brightness = (brightness_range[1] + brightness_range[0]) / 2
        OKhsl = np.c_[
            OKhsl, np.full(OKhsl.shape[0], mid_brightness)]
    else:
        z = emb_fit[:, 2].copy()
        OKhsl[:, 2] = z.copy()
    return OKhsl


# === OKhsl → sRGB (vectorized NumPy port of Björn Ottosson's C++ reference) ===
# These functions convert the OKhsl coordinates into a plottable sRGB array.
# This is a vectorized NumPy port based on the C++ port of the creator of OKlab (MIT Licence, Björn Ottosson, https://bottosson.github.io)

def _srgb_transfer(a):
    return np.where(a <= 0.0031308,
                    12.92 * a,
                    1.055 * np.power(np.clip(a, 0.0, None), 1.0/2.4) - 0.055)


def _oklab_to_linear_srgb(L, a, b):
    l_ = L + 0.3963377774*a + 0.2158037573*b
    m_ = L - 0.1055613458*a - 0.0638541728*b
    s_ = L - 0.0894841775*a - 1.2914855480*b
    l, m, s = l_**3, m_**3, s_**3
    r = +4.0767416621*l - 3.3077115913*m + 0.2309699292*s
    g = -1.2684380046*l + 2.6097574011*m - 0.3413193965*s
    b = -0.0041960863*l - 0.7034186147*m + 1.7076147010*s
    return r, g, b


def _toe_inv(x):
    k1, k2 = 0.206, 0.03
    k3 = (1.0 + k1) / (1.0 + k2)
    return (x*x + k1*x) / (k3*(x + k2))


def _compute_max_saturation(a_, b_):
    cond_r = -1.88170328*a_ - 0.80936493*b_ > 1.0
    cond_g = ~cond_r & (1.81444104*a_ - 1.19445276*b_ > 1.0)
    k0 = np.where(cond_r, 1.19086277,  np.where(
        cond_g,  0.73956515,  1.35733652))
    k1 = np.where(cond_r, 1.76576728,  np.where(
        cond_g, -0.45954404, -0.00915799))
    k2 = np.where(cond_r, 0.59662007,  np.where(
        cond_g,  0.08285427, -1.15130210))
    k3 = np.where(cond_r, 0.75515197,  np.where(
        cond_g,  0.12541070, -0.50559606))
    k4 = np.where(cond_r, 0.56771245,  np.where(
        cond_g,  0.14503204,  0.00692167))
    wl = np.where(cond_r, +4.0767416621,
                  np.where(cond_g, -1.2684380046, -0.0041960863))
    wm = np.where(cond_r, -3.3077115913,
                  np.where(cond_g, +2.6097574011, -0.7034186147))
    ws = np.where(cond_r, +0.2309699292,
                  np.where(cond_g, -0.3413193965, +1.7076147010))
    S = k0 + k1*a_ + k2*b_ + k3*a_**2 + k4*a_*b_
    kl = +0.3963377774*a_ + 0.2158037573*b_
    km = -0.1055613458*a_ - 0.0638541728*b_
    ks = -0.0894841775*a_ - 1.2914855480*b_
    l_ = 1.0 + S*kl
    m_ = 1.0 + S*km
    s_ = 1.0 + S*ks
    l, m, s = l_**3, m_**3, s_**3
    f = wl*l + wm*m + ws*s
    f1 = wl*3*kl*l_**2 + wm*3*km*m_**2 + ws*3*ks*s_**2
    f2 = wl*6*kl**2*l_ + wm*6*km**2*m_ + ws*6*ks**2*s_
    return S - f*f1 / (f1*f1 - 0.5*f*f2)


def _find_cusp(a_, b_):
    Sc = _compute_max_saturation(a_, b_)
    r, g, b = _oklab_to_linear_srgb(1.0, Sc*a_, Sc*b_)
    Lc = np.cbrt(1.0 / np.maximum(np.maximum(r, g), np.maximum(b, 1e-12)))
    return Lc, Lc * Sc


def _find_gamut_intersection(a_, b_, L1, C1, L0, Lc, Cc):
    lower = ((L1 - L0)*Cc - (Lc - L0)*C1) <= 0.0
    t_lo = Cc*L0 / (C1*Lc + Cc*(L0 - L1))
    t_up = Cc*(L0 - 1.0) / (C1*(Lc - 1.0) + Cc*(L0 - L1))
    # One Halley step to refine the upper-half intersection
    kl = +0.3963377774*a_ + 0.2158037573*b_
    km = -0.1055613458*a_ - 0.0638541728*b_
    ks = -0.0894841775*a_ - 1.2914855480*b_
    Lv = L0*(1.0 - t_up) + t_up*L1
    Cv = t_up*C1
    l_ = Lv + Cv*kl
    m_ = Lv + Cv*km
    s_ = Lv + Cv*ks
    l, m, s = l_**3, m_**3, s_**3
    dL = L1 - L0
    l_dt = dL + C1*kl
    m_dt = dL + C1*km
    s_dt = dL + C1*ks
    ldt = 3*l_dt*l_**2
    mdt = 3*m_dt*m_**2
    sdt = 3*s_dt*s_**2
    ldt2 = 6*l_dt**2*l_
    mdt2 = 6*m_dt**2*m_
    sdt2 = 6*s_dt**2*s_

    def _halley(w0, w1, w2):
        f0 = w0*l + w1*m + w2*s - 1.0
        f1 = w0*ldt + w1*mdt + w2*sdt
        f2 = w0*ldt2 + w1*mdt2 + w2*sdt2
        u = f1 / (f1*f1 - 0.5*f0*f2)
        return np.where(u >= 0.0, -f0*u, np.inf)
    dt = np.minimum(
        _halley(+4.0767416621, -3.3077115913, +0.2309699292),
        np.minimum(
            _halley(-1.2684380046, +2.6097574011, -0.3413193965),
            _halley(-0.0041960863, -0.7034186147, +1.7076147010),
        )
    )
    return np.where(lower, t_lo, t_up + dt)


def _get_ST_mid(a_, b_):
    S = 0.11516993 + 1.0 / (
        +7.44778970 + 4.15901240*b_
        + a_*(-2.19557347 + 1.75198401*b_
              + a_*(-2.13704948 - 10.02301043*b_
                    + a_*(-4.24894561 + 5.38770819*b_ + 4.69891013*a_))))
    T = 0.11239642 + 1.0 / (
        +1.61320320 - 0.68124379*b_
        + a_*(+0.40370612 + 0.90148123*b_
              + a_*(-0.27087943 + 0.61223990*b_
                    + a_*(+0.00299215 - 0.45399568*b_ - 0.14661872*a_))))
    return S, T


def _get_Cs(L, a_, b_):
    Lc, Cc = _find_cusp(a_, b_)
    C_max = _find_gamut_intersection(a_, b_, L, 1.0, L, Lc, Cc)
    S_max = Cc / Lc
    T_max = Cc / (1.0 - Lc)
    k = C_max / np.minimum(L*S_max, (1.0 - L)*T_max)
    S_mid, T_mid = _get_ST_mid(a_, b_)
    Ca = L*S_mid
    Cb = (1.0 - L)*T_mid
    C_mid = 0.9 * k * (1.0 / (1.0/Ca**4 + 1.0/Cb**4))**0.25
    Ca = L*0.4
    Cb = (1.0 - L)*0.8
    C_0 = (1.0 / (1.0/Ca**2 + 1.0/Cb**2))**0.5
    return C_0, C_mid, C_max


def _okhsl_to_srgb_array(h_deg, s, l):
    """OKhsl → sRGB (vectorized). h_deg in degrees (any range), s/l in [0, 1].
    Returns float64 array shape (n, 3), NOT yet clipped (for gamut reporting)."""
    h_deg = np.asarray(h_deg, dtype=float)
    s = np.asarray(s,     dtype=float)
    l = np.asarray(l,     dtype=float)
    rgb = np.empty((len(l), 3), dtype=float)

    white = l >= 1.0
    black = l <= 0.0
    inner = ~(white | black)
    rgb[white] = 1.0
    rgb[black] = 0.0

    if inner.any():
        h = (h_deg[inner] % 360.0) / 360.0
        sv = s[inner]
        lv = l[inner]
        a_ = np.cos(2*np.pi*h)
        b_ = np.sin(2*np.pi*h)
        L = _toe_inv(lv)

        C_0, C_mid, C_max = _get_Cs(L, a_, b_)

        MID, MID_INV = 0.8, 1.25
        lo = sv < MID

        # Low-saturation segment (s < 0.8): dC/ds|s=0 = C_0, C(0.8) = C_mid
        k1 = MID * C_0
        k2 = 1.0 - k1/C_mid
        t = MID_INV * sv
        C_lo = t * k1 / (1.0 - k2*t)

        # High-saturation segment (s >= 0.8): C(0.8) = C_mid, C(1.0) = C_max
        k1h = (1.0 - MID) * C_mid**2 * MID_INV**2 / C_0
        k2h = 1.0 - k1h / (C_max - C_mid)
        th = (sv - MID) / (1.0 - MID)
        C_hi = C_mid + th * k1h / (1.0 - k2h*th)

        C = np.where(lo, C_lo, C_hi)
        r, g, b = _oklab_to_linear_srgb(L, C*a_, C*b_)
        rgb[inner] = np.stack(
            [_srgb_transfer(r), _srgb_transfer(g), _srgb_transfer(b)], axis=1)

        # Report gamut clipping before applying it
        n_clipped = ((rgb < 0) | (rgb > 1)).any(axis=1).sum()
        print(f"Cells outside sRGB gamut (clipped): {n_clipped} of {len(rgb)} "
              f"({100 * n_clipped / len(rgb):.1f}%)")

        rgb = np.clip(rgb, 0.0, 1.0)

    return rgb


# === OKhsv → sRGB (vectorized NumPy port of Björn Ottosson's C++ reference) ===
# OKhsv is a similar colorspace by Björn Ottosson, not currently used by clrmappy

def okhsv_to_srgb_array(h_deg, s, v):
    """OKhsv → sRGB (vectorized). h_deg in degrees (any range), s/v in [0, 1].
    Returns float64 array shape (n, 3), NOT yet clipped (for gamut reporting)."""
    h_deg = np.asarray(h_deg, dtype=float)
    s = np.asarray(s,     dtype=float)
    v = np.asarray(v,     dtype=float)
    rgb = np.empty((len(v), 3), dtype=float)

    black = v <= 0.0
    inner = ~black
    rgb[black] = 0.0

    if inner.any():
        h = (h_deg[inner] % 360.0) / 360.0
        sv = s[inner]
        vv = v[inner]

        a_ = np.cos(2 * np.pi * h)
        b_ = np.sin(2 * np.pi * h)

        Lc, Cc = _find_cusp(a_, b_)
        S_max = Cc / Lc
        T_max = Cc / (1.0 - Lc)
        S_0 = 0.5
        k = 1.0 - S_0 / S_max

        denom = S_0 + T_max - T_max * k * sv
        L_v = 1.0 - sv * S_0 / denom
        C_v = sv * T_max * S_0 / denom

        L = vv * L_v
        C = vv * C_v

        L_vt = _toe_inv(L_v)
        C_vt = C_v * L_vt / L_v

        L_new = _toe_inv(L)
        C = np.where(L > 0, C * L_new / L, 0.0)
        L = L_new

        rs, gs, bs = _oklab_to_linear_srgb(L_vt, a_ * C_vt, b_ * C_vt)
        scale_L = np.cbrt(
            1.0 / np.maximum(np.maximum(rs, gs), np.maximum(bs, 0.0)))

        L = L * scale_L
        C = C * scale_L

        r, g, b = _oklab_to_linear_srgb(L, C * a_, C * b_)
        rgb[inner] = np.stack(
            [_srgb_transfer(r), _srgb_transfer(g), _srgb_transfer(b)], axis=1)

    return rgb
