from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Internal OKhsl helpers — relative import so this module works when used
# as part of the ``clrmappy`` package (i.e. ``from clrmappy import ...``).
# The leading dot tells Python to look in the SAME package (folder
# containing __init__.py), not in sys.path.
from ._okhsl_utils import (
    _fit_and_center_2d3d_pca,
    _apply_rotation_scaling,
    _saturation_enhancement,
    _emb_to_OKhsl,
    _okhsl_to_srgb_array,
    _srgb_transfer,
)


'''
These are the main functions of clrmappy. The 3 main functions are: emb_to_rgb, emb_to_okhsl and emb_to_cielab.
'''

# === RGB PIPELINE ===


def emb_to_rgb(emb_3d, equal_variance_mode=False):
    """
    This function calculates an unsupervised RGB coloring based on a 3D embedding of any kind.
    The 3D Data are min-max scaled to the cubic RGB color space. The x-axis encodes r values,
    the y-axis encodes g values, the z-axis encodes b values (every axis ranges from 0-1).
    The RGB color space is simple, but not perceptually uniform.

    Parameters
    ----------
    emb_3d : numpy array
        This is the 3D embedding, which the coloring of each data point is based on (e.g. UMAP, t-SNE, PCA, ...)
        The 3D data themselves can be colored using the rgb colors, that this function calculates.
        The colors can of course be also used to color the same data set in 2D after dimension reduction.
    equal_variance_mode : bool
        This turns a mode on/off, which turns the 3D data in a way, that all axes explain rougly the same/similar
        amount of variance in the data. This is achieved by calculating a 3D PCA on the 3D data and then rotating
        by 45° around every axis.

    Returns
    -----
    clrmappy : rgb array; each "point"/line is assigned a rgb value
    emb3d_fit : rgb array; This is how the provided 3d embedding looks like after min-max scaling (and PCA + rotation, if appliedd).
    """
    # Verify the dimensionality of the provided array. Only 3 dimensional is valid!
    if emb_3d.shape[1] == 3:
        print("The user has provided a valid 3D Embedding.")

        emb3d_fit = emb_3d.copy()

        if equal_variance_mode == True:
            # Rotation Matrix, that if applied to a PCA turns by 45° around each axis -> every axis involves every PCA-component!
            rotation_matrix = np.array([
                [0.500,  0.500, -0.707],
                [-0.146,  0.854,  0.500],
                [0.854, -0.146,  0.500],
            ])

            # 3D PCA on 3D UMAP
            pca_3d = PCA(n_components=3)
            emb3d_fit = pca_3d.fit_transform(emb_3d)

            # Here the Rotation Matrix is applied to the fitted embedding
            emb3d_fit = emb3d_fit @ rotation_matrix

        # Min-Max-Scaling (0-1) - vectorized (scales each axis (x,y,z to 0-1))
        emb3d_fit = (emb3d_fit - emb3d_fit.min(axis=0)) / \
            (emb3d_fit.max(axis=0) - emb3d_fit.min(axis=0))

        clrmappy = emb3d_fit.copy()

        return {
            'clrmappy': clrmappy,
            'emb3d_fit': emb3d_fit,
        }
    else:
        print("ERROR: The provided embedding does not have the valid dimensionality of 3.")

# === OKhsl PIPELINE ===


def emb_to_okhsl(emb_3d=None, emb_2d=None, iso_rot_scale=True, equal_variance_mode=False, pc1_and_2_from_2d=False, brightness_range=[0.2, 0.8], saturation_enhancement=True, saturation_range=[0.1, 1.0], center_around='mid'):
    """
    This function calculates an unsupervised OKhsl coloring (coloring within the OKhsl colorspace) based on a 3D embedding and/or 2D embedding of any kind.
    The 3D Data are fitted with in the cylindrical color space, where the xy-angle of each vector encodes the hue (0° - 360°), 
    the magnitude of the xy-component of each vector (distance from the center/ z-axis) encodes the saturation (0 - 1) and the z value encodes the brightness (0 - 1).

    The OKhsl colorspace is a colorspace by Björn Ottosson based his perceptual uniform color space OKlab (https://bottosson.github.io).
    The OKhsl maps OKlab to a simple, cylindrical sRGB gamut, which makes it ideal for fast, simple unsupervised coloring.

    The user can provide only a 3D embedding, only a 2D embedding or both. If only a 2D embedding is provided 
    the brightness of the coloring (z axis) is set to a constant value.

    Parameters
    ----------
    emb_3d, emb_2d : numpy array
        These are the 3D embeddings, which the coloring of each data point is based on (e.g. UMAP, t-SNE, PCA, ...)
        The user chooses with the following parameters how these embeddings are used to calculate the final clrmappy_array

    rot_scaling_algo : bool
        This turns an optimization algorithm on/off, which performs a 3D PCA on the 3D embedding 
        in order to rotate the axis of greatest variance (PC1) around PC2 to find the optimal rotation angle 
        for a maximum mean distance of the points from the z axis, while scaling the embedding to the height 
        of the OKhsl-cylinder (height 1). It essentially scales and rotates the embedding to find the highest mean saturation, 
        without distorting the data. This also minimizes distortion of the data, if the user chooses to min-max-scale/increase 
        the saturation manually anyway.

    equal_variance_mode : bool
        This turns a mode on/off, which turns the 3D data in a way, that all axes explain rougly the same/similar
        amount of variance in the data. This is achieved by calculating a 3D PCA on the 3D data and then rotating
        by 45° around every axis.

    pc1_and_2_from_2d  : bool
        This turns a mode on/off that performs a 2D PCA on the provided 2d embedding and a 3D PCA on the provided 3D embedding
        and calculates the the hue/saturation for the coloring from the pc1 and 2 from the 2D embedding. The brightess is calculated
        from the pc3 of the 3D embedding. This is useful in some cases, in which the 2D embeddings look very different compared to the 
        3D embeddings, but the 2D embedding is still the one used for clustering and classification. Compared to just coloring from
        the 2D embedding with constant brightness, this provides extra structure and depth, that helps analysing the data.
        This requires input of a 2D and a 3D embedding of the same data with the same parameters (e.g 2D and 3D umap of a single cell dataset).

    brightness_range : list
        This is the brightness range used for the coloring. It makes sense to exclude very low or high values, as those are hard to plot and to perceive, 
        especially for perceiving differences .

    saturation_enhancement : bool
        This sets saturation_enhancement on/off. If off, the coloring might seem very muted. If on, the saturation range is used for min-max scaling

    saturation_range : bool
        This range is used for min-max scaling of the saturation. It makes sense to stretch the saturation to 1 and avoid very low saturation values (gray).

    center_around : string
        This is parameter, that chooses how the fitted embedding is centered, either around 'mid' or 'mean'.


    Returns
    -----
    clrmappy : rgb array; each "point"/line is assigned a rgb value
    emb3d_fit : rgb array; This is how the provided 3d embedding looks like after min-max scaling (and PCA + rotation, if appliedd).
    """

    # 3D Pipeline -> Coloring is based on the provided 3D Embedding

    if emb_3d is not None:  # -> The user provided a 3D Embedding

        if emb_3d.shape[1] == 3:
            print("The user has provided a valid 3D Embedding.")
        else:
            print(
                "ERROR: The 3D Embedding does not have a valid dimensionality (not equal to 3).")
            return

        if emb_2d is None:  # -> The user provided only a 3D Embedding and not a 2D Embedding
            pc1_and_2_from_2d = False
            # pc1 and 2 cannot be calculated from 2D

        if emb_2d is not None:  # -> The user provided a 2D Embedding alongside the 3D Embedding
            if emb_2d.shape[1] == 2:
                print(
                    "The user has provided a valid 2D Embedding alongside a valid 3D Embedding.")
            else:
                print(
                    "ERROR: The 2D Embedding does not have a valid dimensionality (not equal to 2).")
                return

        # 3D PCA on 3D Embedding
        emb3d_fit = _fit_and_center_2d3d_pca(
            emb_3d=emb_3d, emb_2d=emb_2d, equal_variance_mode=equal_variance_mode, pc1_and_2_from_2d=pc1_and_2_from_2d, center_around=center_around)

        # Rotation Scaling algorithm
        if iso_rot_scale == True:
            result_rot = _apply_rotation_scaling(
                emb3d_fit=emb3d_fit, brightness_range=brightness_range)
            emb3d_fit = result_rot['embedding'].copy()
            r_z = result_rot['r_z'].copy()

            if saturation_enhancement == True:
                result_sat = _saturation_enhancement(
                    emb3d_fit, saturation_range)
                emb3d_fit = result_sat['embedding'].copy()
                r_z = result_sat['r_z'].copy()

        # simple min max scaling, instead of rotation optimization algorithm
        elif iso_rot_scale == False:
            # distance from z-axis after rotation
            r_z = np.sqrt(emb3d_fit[:, 0]**2 + emb3d_fit[:, 1]**2)

            # scale x axis and y axis stored in emb3d_fit according to r_z.max() (so the maximum r becomes 1)
            emb3d_fit[:, :2] /= r_z.max()
            r_z /= r_z.max()

            # min-max scaling for z-values with brightness range
            z = emb3d_fit[:, 2]
            z_min, z_max = z.min(), z.max()
            z_scaled_min, z_scaled_max = brightness_range
            emb3d_fit[:, 2] = (z - z_min) / (z_max - z_min) * \
                (z_scaled_max - z_scaled_min) + z_scaled_min

            if saturation_enhancement == True:
                result_sat = _saturation_enhancement(
                    emb_fit=emb3d_fit, saturation_range=saturation_range)
                emb3d_fit = result_sat['embedding'].copy()
                r_z = result_sat['r_z'].copy()

        # convert to OKhsl
        OKhsl = _emb_to_OKhsl(emb_fit=emb3d_fit, r_z=r_z,
                              brightness_range=brightness_range)

        # Convert OKhsl to rgb array
        clrmappy_array = _okhsl_to_srgb_array(
            OKhsl[:, 0], OKhsl[:, 1], OKhsl[:, 2])

        return {
            'clrmappy': clrmappy_array,
            'emb_fit': emb3d_fit,
        }

    # 2D PIPELINE

    elif emb_3d is None and emb_2d is not None:
        if emb_2d.shape[1] == 2:
            print("The user has provided a valid 2D UMAP.")

            # 2D PCA on 2D UMAP
            emb2d_fit = _fit_and_center_2d3d_pca(
                emb_3d=None, emb_2d=emb_2d,  equal_variance_mode=equal_variance_mode, pc1_and_2_from_2d=pc1_and_2_from_2d, center_around=center_around)

            # No rotation scaling, only the scaling of r_max here (see function _apply_rotation_scaling)
            # distance from z-axis after rotation
            r = np.sqrt(emb2d_fit[:, 0]**2 + emb2d_fit[:, 1]**2)

            # scale x axis and y axis stored in emb2d_fit according to r_z.max() (so the maximum r becomes 1)
            emb2d_fit[:, :2] /= r.max()
            r /= r.max()

            if saturation_enhancement == True:
                result_sat = _saturation_enhancement(
                    emb2d_fit, saturation_range)
                emb2d_fit = result_sat['embedding'].copy()
                r = result_sat['r_z'].copy()

            # convert to OKhsl
            OKhsl = _emb_to_OKhsl(emb_fit=emb2d_fit, r_z=r,
                                  brightness_range=brightness_range)

            # Convert OKhsl to rgb array
            clrmappy = _okhsl_to_srgb_array(
                OKhsl[:, 0], OKhsl[:, 1], OKhsl[:, 2])

            return {
                'clrmappy': clrmappy,
                'emb_fit': emb3d_fit,
            }

        else:
            print(
                "ERROR: The 2D UMAP does not have a valid dimensionality (not equal to 2).")
            return

    else:
        print('ERROR: There has been an error with the umap files. Please provide a 2D UMAP, a 3D UMAP or a 3D UMAP AND 2D UMAP.')
        return

# === U-CIE (CIELab) PIPELINE ===
# Python interface to the U-CIE R package ((MIT License, Copyright (c) 2021 ucie authors, github.com/mikelkou/ucie).
# Algorithm: fits the 3D UMAP convex hull into the CIELab color gamut via
# Nelder-Mead optimisation (scale + 3-axis rotation + translation),
# then converts CIELab coordinates to sRGB.


def emb_to_cielab(embedding, WL=1.0, Wa=1.0, Wb=1.0, S=1.0):
    """
    This function calculates an unsupervised CIElab coloring based on a 3D embedding of any kind by using the U-CIE R package.

    U-CIE optimises a rigid-body transform (scale, 3-axis rotation,
    translation) to fit the UMAP point cloud inside the CIELab gamut,
    then converts the resulting CIELab coordinates to sRGB.
    The R process runs headless (RGL_USE_NULL=TRUE).

    Parameters
    ----------
    WL, Wa, Wb : float
        Per-axis weights for the L*, a*, b* dimensions (default 1).
    S : float
        Global post-fit scaling factor (default 1).

    Returns
    -----
    clrmappy : rgb array; each "point"/line is assigned a rgb value
    """
    import subprocess
    import tempfile
    import os

    _ensure_ucie_installed()

    embedding  # has to be 3d!!!

    with tempfile.TemporaryDirectory() as tmpdir:
        in_csv = os.path.join(tmpdir, 'umap.csv')
        out_csv = os.path.join(tmpdir, 'lab.csv')
        r_script = os.path.join(tmpdir, 'run_ucie.R')

        np.savetxt(in_csv, embedding, delimiter=',',
                   header='V1,V2,V3', comments='')

        with open(r_script, 'w') as f:
            f.write(_UCIE_R)

        env = os.environ.copy()
        env['RGL_USE_NULL'] = 'TRUE'    # headless rgl — no display needed

        proc = subprocess.run(
            ['Rscript', '--vanilla', r_script,
             in_csv, out_csv, str(WL), str(Wa), str(Wb), str(S)],
            capture_output=True, text=True, env=env,
        )

        if proc.returncode != 0:
            raise RuntimeError(f"U-CIE R process failed:\n{proc.stderr}")
        if proc.stderr.strip():
            print(proc.stderr.strip())   # forward R messages / warnings

        lab = np.loadtxt(out_csv, delimiter=',', skiprows=1)

    # I am not currently using the adata object within the functions
    # adata.obsm['X_umap_ucie_lab'] = lab
    # print("Saved CIELab array in adata.obsm['X_umap_ucie_lab'].")

    clrmappy = np.clip(_lab_to_srgb(lab), 0.0, 1.0)
    # adata.obsm['X_umap_ucie_rgb'] = rgb_colors
    # print("Saved sRGB array in adata.obsm['X_umap_ucie_rgb'].")

    return clrmappy


_UCIE_R = (
    "suppressPackageStartupMessages(library(ucie))\n"
    "args    <- commandArgs(trailingOnly = TRUE)\n"
    "in_csv  <- args[1]\n"
    "out_csv <- args[2]\n"
    "WL      <- as.numeric(args[3])\n"
    "Wa      <- as.numeric(args[4])\n"
    "Wb      <- as.numeric(args[5])\n"
    "S       <- as.numeric(args[6])\n"
    "data    <- read.csv(in_csv)\n"
    "colors  <- data2cielab(data, WL = WL, Wa = Wa, Wb = Wb, S = S,\n"
    "                       LAB_coordinates = TRUE)\n"
    'write.csv(colors[, c("L", "a", "b")], out_csv, row.names = FALSE)\n'
)


def _lab_to_srgb(lab):
    """CIELab (D65) → sRGB, shape (n, 3). Reuses _srgb_transfer for gamma."""
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    delta = 6.0 / 29.0

    def _f_inv(t):
        return np.where(t > delta, t ** 3, 3.0 * delta**2 * (t - 4.0 / 29.0))

    X = _f_inv(fx) * 0.95046
    Y = _f_inv(fy)
    Z = _f_inv(fz) * 1.08906
    r = 3.2404542*X - 1.5371385*Y - 0.4985314*Z
    g = -0.9692660*X + 1.8760108*Y + 0.0415560*Z
    b_ = 0.0556434*X - 0.2040259*Y + 1.0572252*Z
    return np.stack([_srgb_transfer(r), _srgb_transfer(g), _srgb_transfer(b_)], axis=1)


def _ensure_ucie_installed():
    """Install the ucie R package from GitHub if not already present."""
    import subprocess
    import os
    check = subprocess.run(
        ['Rscript', '--vanilla', '-e',
         'if (!requireNamespace("ucie", quietly=TRUE)) stop("missing")'],
        capture_output=True, text=True,
    )
    if check.returncode == 0:
        return
    print("ucie not found — installing from mikelkou/ucie ...")
    r_code = (
        "Sys.unsetenv(c('GITHUB_PAT', 'GITHUB_TOKEN', 'GH_TOKEN'));"
        "if (!requireNamespace('remotes', quietly=TRUE))"
        "    install.packages('remotes', repos='https://cloud.r-project.org');"
        "remotes::install_github('mikelkou/ucie', upgrade='never');"
        "if (!requireNamespace('ucie', quietly=TRUE)) stop('ucie not installed after install_github')"
    )
    # Strip GitHub auth from the subprocess environment so R never sees a bad token.
    clean_env = {k: v for k, v in os.environ.items()
                 if k not in ('GITHUB_PAT', 'GITHUB_TOKEN', 'GH_TOKEN')}
    inst = subprocess.run(['Rscript', '--vanilla', '-e', r_code],
                          capture_output=True, text=True, env=clean_env)
    if inst.returncode != 0 or 'ucie not installed' in inst.stderr:
        raise RuntimeError(
            "ucie installation failed. Ensure rgl and its system dependencies "
            f"are available (on macOS: XQuartz).\n"
            f"--- stdout ---\n{inst.stdout}\n"
            f"--- stderr ---\n{inst.stderr}"
        )
    print("ucie installed successfully.")
