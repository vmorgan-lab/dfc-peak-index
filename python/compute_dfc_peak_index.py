# This script computes null threshold maps for dFC Peak Index based on the manuscript by Sainburg et al., Epilepsia 2026: 
# The dynamic functional peak index: detection of interictal epileptic activity with fMRI

# L Sainburg May 2026
# Adapted from MATLAB scripts at https://github.com/vmorgan-lab/dfc-peak-index/

"""Compute seed-coupled dFC peak index maps from 4D NIfTI files.

This module is intentionally agnostic to BIDS or any other dataset layout.
Input 4D NIfTI files are supplied directly as full paths. Dataset-specific
file discovery and organization can be handled by a separate wrapper script.

Thresholding options
--------------------
Provide exactly one of:

1. A 3D null-threshold map (voxelwise thresholding), or
2. A constant threshold value (applied to all voxels).

Timing
------
TR is read from each input NIfTI header.

Output naming
-------------
The input filename is preserved except for the NIfTI/BOLD suffix, which is
replaced with ``_dfc-peak-index[_{seed_name}].nii.gz``.

Examples
--------
Single input:

    python dfc_peak_index.py \
        --nifti /full/path/run1_bold.nii.gz \
        --output-dir /full/path/output \
        --dmn-mask /full/path/dmn_mask.nii.gz \
        --threshold-value -0.5

Many inputs from a CSV (one full NIfTI path per row):

    python dfc_peak_index.py \
        --nifti-csv /full/path/nifti_paths.csv \
        --output-dir /full/path/output \
        --dmn-mask /full/path/dmn_mask.nii.gz \
        --null-threshold-map /full/path/threshold_map.nii.gz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn.image import load_img, new_img_like
from scipy.signal import find_peaks


def _read_nifti_csv(csv_path: str | Path) -> list[Path]:
    """Read full NIfTI paths from the first column of a CSV file."""
    csv_path = Path(csv_path)

    if not csv_path.is_file():
        raise FileNotFoundError(f"NIfTI CSV does not exist: {csv_path}")

    paths: list[Path] = []

    with csv_path.open("r", encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, start=1):
            value = line.strip()

            if not value:
                continue

            # Use the first column if the file contains additional columns.
            value = value.split(",", 1)[0].strip().strip('"').strip("'")

            # Allow an optional header in the first row.
            if line_number == 1 and value.lower() in {
                "nifti",
                "nifti_path",
                "nifti_paths",
                "path",
                "file",
                "filename",
            }:
                continue

            if value:
                paths.append(Path(value).expanduser())

    if not paths:
        raise ValueError(
            f"No NIfTI paths were found in the first column of: {csv_path}"
        )

    return paths


def _load_4d_data(
    nifti_path: str | Path,
) -> tuple[nib.Nifti1Image, np.ndarray]:
    img = load_img(str(nifti_path))
    data = img.get_fdata(dtype=np.float64)

    if data.ndim != 4:
        raise ValueError(
            f"Expected a 4D fMRI image, got shape {data.shape} "
            f"for {nifti_path}"
        )

    return img, data


def _load_3d_mask(
    mask_path: str | Path,
) -> tuple[nib.Nifti1Image, np.ndarray]:
    img = load_img(str(mask_path))
    data = img.get_fdata(dtype=np.float64)

    if data.ndim != 3:
        raise ValueError(
            f"Expected a 3D mask, got shape {data.shape} for {mask_path}"
        )

    return img, data > 0


def _get_tr_from_img(
    img: nib.Nifti1Image,
    nifti_path: str | Path,
) -> float:
    zooms = img.header.get_zooms()

    if len(zooms) < 4 or not np.isfinite(zooms[3]) or zooms[3] <= 0:
        raise ValueError(
            f"Could not read a valid TR from NIfTI header for {nifti_path}"
        )

    return float(zooms[3])


def _zscore_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.full_like(x, np.nan, dtype=np.float64)

    good = np.isfinite(x)
    if good.sum() < 2:
        return out

    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=1)

    if not np.isfinite(sd) or sd == 0:
        out[good] = 0.0
        return out

    out[good] = (x[good] - mu) / sd
    return out


def _zscore_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)

    if x.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {x.shape}")

    mean = np.nanmean(x, axis=1, keepdims=True)
    std = np.nanstd(x, axis=1, ddof=1, keepdims=True)

    out = np.full_like(x, np.nan, dtype=np.float64)
    good = np.isfinite(x)
    valid_rows = np.isfinite(std[:, 0]) & (std[:, 0] != 0)

    if np.any(valid_rows):
        out[valid_rows] = (
            x[valid_rows] - mean[valid_rows]
        ) / std[valid_rows]
        out[~good] = np.nan

    if np.any(~valid_rows):
        out[~valid_rows] = 0.0
        out[~good & ~valid_rows[:, None]] = np.nan

    return out


def _compute_dfc_matrix(
    data_4d: np.ndarray,
    dmn_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if data_4d.ndim != 4:
        raise ValueError("data_4d must be 4D")

    if dmn_mask.shape != data_4d.shape[:3]:
        raise ValueError(
            f"DMN mask shape {dmn_mask.shape} does not match "
            f"data spatial shape {data_4d.shape[:3]}"
        )

    n_vox = int(np.prod(data_4d.shape[:3]))
    n_time = data_4d.shape[3]

    data_vx = data_4d.reshape(n_vox, n_time)
    dmn_vx = dmn_mask.reshape(-1)

    if not np.any(dmn_vx):
        raise ValueError("DMN mask contains no nonzero voxels")

    dmn_ts = np.nanmean(data_vx[dmn_vx, :], axis=0)

    data_vx_norm = _zscore_rows(data_vx)
    dmn_ts_norm = _zscore_1d(dmn_ts)

    dfc = data_vx_norm * dmn_ts_norm[None, :]

    return dfc, dmn_ts


def _save_3d_map(
    reference_img: nib.Nifti1Image,
    data_3d: np.ndarray,
    out_path: str | Path,
) -> str:
    out_path = str(out_path)

    img = new_img_like(
        reference_img,
        data_3d.astype(np.float32),
        copy_header=True,
    )

    nib.save(img, out_path)

    return out_path


def _output_peak_filename(
    input_path: Path,
    dfc_seed_name: str | None = None,
) -> str:
    """Generate a peak-index filename from an input NIfTI filename."""
    name = input_path.name

    if name.endswith("_bold.nii.gz"):
        stem = name[:-len("_bold.nii.gz")]
    elif name.endswith("_bold.nii"):
        stem = name[:-len("_bold.nii")]
    elif name.endswith(".nii.gz"):
        stem = name[:-len(".nii.gz")]
    elif name.endswith(".nii"):
        stem = name[:-len(".nii")]
    else:
        raise ValueError(
            f"Expected a NIfTI input file (.nii or .nii.gz), got: {name}"
        )

    seed_tag = f"_{dfc_seed_name}" if dfc_seed_name else ""

    return f"{stem}_dfc-peak-index{seed_tag}.nii.gz"


def _find_control_negative_peaks(
    ts: np.ndarray,
    thr_v: float,
) -> np.ndarray:
    # Peaks in -dFC are local minima in dFC.
    peaks, _ = find_peaks(-ts, height=-thr_v)
    return peaks


def _prepare_thresholds(
    threshold_map_path: str | Path | None,
    threshold_value: float | None,
) -> tuple[np.ndarray | float, str]:
    if (threshold_map_path is None) == (threshold_value is None):
        raise ValueError(
            "Provide exactly one of threshold_map_path or threshold_value"
        )

    if threshold_map_path is not None:
        thr_img = load_img(str(threshold_map_path))
        thr = thr_img.get_fdata(dtype=np.float64)

        if thr.ndim != 3:
            raise ValueError(
                f"Expected a 3D threshold map, got shape {thr.shape}"
            )

        return thr, "map"

    assert threshold_value is not None

    if not np.isfinite(threshold_value):
        raise ValueError("threshold_value must be finite")

    return float(threshold_value), "constant"


def compute_dfc_peak_index(
    nifti_paths: list[str | Path] | tuple[str | Path, ...] | str | Path,
    output_dir: str | Path,
    dmn_mask_path: str | Path,
    threshold_map_path: str | Path | None = None,
    threshold_value: float | None = None,
    dfc_seed_name: str | None = None,
) -> list[str]:
    """Compute seed-coupled peak index maps for one or more NIfTIs.

    Parameters
    ----------
    nifti_paths:
        Full path to one 4D NIfTI, or a list/tuple of full NIfTI paths.
    output_dir:
        Directory where output peak-index maps will be written.
    dmn_mask_path:
        Full path to the 3D seed/DMN mask NIfTI.
    threshold_map_path:
        Optional full path to a 3D voxelwise null-threshold map.
    threshold_value:
        Optional constant threshold applied to all voxels.
    dfc_seed_name:
        Optional seed name appended to output filenames.

    Returns
    -------
    list[str]
        Full paths to the generated peak-index maps.
    """
    if isinstance(nifti_paths, (str, Path)):
        nifti_paths = [nifti_paths]

    nifti_paths = [Path(path) for path in nifti_paths]

    if not nifti_paths:
        raise ValueError("No input NIfTI paths were provided")

    for nifti_path in nifti_paths:
        if not nifti_path.is_file():
            raise FileNotFoundError(
                f"Input NIfTI does not exist: {nifti_path}"
            )

    dmn_mask_path = Path(dmn_mask_path)
    if not dmn_mask_path.is_file():
        raise FileNotFoundError(
            f"DMN mask does not exist: {dmn_mask_path}"
        )

    if threshold_map_path is not None:
        threshold_map_path = Path(threshold_map_path)
        if not threshold_map_path.is_file():
            raise FileNotFoundError(
                f"Threshold map does not exist: {threshold_map_path}"
            )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    threshold_field, threshold_mode = _prepare_thresholds(
        threshold_map_path,
        threshold_value,
    )

    _, dmn_mask = _load_3d_mask(dmn_mask_path)

    out_paths: list[str] = []

    for i, nifti_path in enumerate(nifti_paths, start=1):
        print(
            f"Processing {i}/{len(nifti_paths)}: {nifti_path}"
        )

        img, data = _load_4d_data(nifti_path)
        tr = _get_tr_from_img(img, nifti_path)

        if data.shape[:3] != dmn_mask.shape:
            raise ValueError(
                f"DMN mask shape {dmn_mask.shape} does not match "
                f"subject data shape {data.shape[:3]} for {nifti_path}"
            )

        if threshold_mode == "map":
            thr = np.asarray(
                threshold_field,
                dtype=np.float64,
            )

            if thr.shape != data.shape[:3]:
                raise ValueError(
                    f"Threshold map shape {thr.shape} does not match "
                    f"subject data shape {data.shape[:3]} "
                    f"for {nifti_path}"
                )

            thr_flat = thr.reshape(-1)

        else:
            thr_flat = float(threshold_field)

        dfc, dmn_ts = _compute_dfc_matrix(
            data,
            dmn_mask,
        )

        n_vox, n_time = dfc.shape
        dmn_negative = np.where(dmn_ts < 0)[0]

        peak_freq = np.full(
            n_vox,
            np.nan,
            dtype=np.float64,
        )

        for v in range(n_vox):
            ts = dfc[v, :]

            if not np.all(np.isfinite(ts)):
                continue

            if threshold_mode == "map":
                thr_v = thr_flat[v]
            else:
                thr_v = thr_flat

            if not np.isfinite(thr_v):
                continue

            peaks = _find_control_negative_peaks(
                ts,
                float(thr_v),
            )

            if peaks.size == 0 or dmn_negative.size == 0:
                peak_freq[v] = 0.0
                continue

            count = np.intersect1d(
                peaks,
                dmn_negative,
                assume_unique=False,
            ).size

            peak_freq[v] = count / (tr * n_time)

        out_map = peak_freq.reshape(data.shape[:3])

        out_name = _output_peak_filename(
            nifti_path,
            dfc_seed_name=dfc_seed_name,
        )

        out_path = output_dir / out_name

        out_paths.append(
            _save_3d_map(
                img,
                out_map,
                out_path,
            )
        )

        print(f"Saved: {out_path}")

    return out_paths


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)

    input_group.add_argument(
        "--nifti",
        dest="nifti_paths",
        action="append",
        help=(
            "Full path to an input 4D NIfTI. "
            "Repeat --nifti to process multiple files."
        ),
    )

    input_group.add_argument(
        "--nifti-csv",
        dest="nifti_csv",
        help=(
            "CSV containing full NIfTI paths in the first column. "
            "An optional header is allowed."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where output NIfTI files will be written.",
    )

    parser.add_argument(
        "--dmn-mask",
        required=True,
        help="Full path to the 3D DMN/seed mask NIfTI.",
    )

    parser.add_argument(
        "--dfc-seed-name",
        default=None,
        help="Optional seed name appended to output filenames.",
    )

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--null-threshold-map",
        dest="threshold_map_path",
        help="Full path to a 3D null-threshold map.",
    )

    group.add_argument(
        "--threshold-value",
        dest="threshold_value",
        type=float,
        help="Constant threshold applied to all voxels.",
    )

    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.nifti_csv is not None:
        nifti_paths = _read_nifti_csv(args.nifti_csv)
    else:
        nifti_paths = args.nifti_paths

    compute_dfc_peak_index(
        nifti_paths=nifti_paths,
        output_dir=args.output_dir,
        dmn_mask_path=args.dmn_mask,
        threshold_map_path=args.threshold_map_path,
        threshold_value=args.threshold_value,
        dfc_seed_name=args.dfc_seed_name,
    )


if __name__ == "__main__":
    main()


__all__ = ["compute_dfc_peak_index"]
