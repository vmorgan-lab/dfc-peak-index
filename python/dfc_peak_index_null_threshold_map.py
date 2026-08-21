# This script computes null threshold maps for dFC Peak Index based on the
# manuscript by Sainburg et al., Epilepsia 2026:
#
# The dynamic functional peak index: detection of interictal epileptic activity
# with fMRI
#
# L Sainburg May 2026
#
# Adapted from MATLAB scripts at:
# https://github.com/vmorgan-lab/dfc-peak-index/

"""Build a null-threshold map from control DMN dFC 5th-percentile maps.

This script is intentionally agnostic to BIDS or any other dataset layout.
It reads a CSV containing full paths to control 4D NIfTI files, computes a
5th-percentile DMN-coupled dFC map for each input file, and writes a voxelwise
median null-threshold map across all control maps.

The CSV should contain one full NIfTI path per row in the first column. An
optional header such as ``nifti_path`` is allowed.

Example CSV
-----------
nifti_path
/full/path/control01_bold.nii.gz
/full/path/control02_bold.nii.gz
/full/path/control03_bold.nii.gz

Usage
-----
python dfc_peak_index_null_threshold_map.py \
    --nifti-csv /full/path/control_niftis.csv \
    --dmn-mask /full/path/dmn_mask.nii.gz \
    --output-dir /full/path/output
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn.image import load_img, new_img_like


@dataclass(frozen=True)
class DFCResult:
    """Paths to generated individual and group-level outputs."""

    individual_map_paths: list[str]
    median_map_path: str | None = None


def _read_nifti_csv(csv_path: str | Path) -> list[Path]:
    """Read full NIfTI paths from the first column of a CSV file."""
    csv_path = Path(csv_path)

    if not csv_path.is_file():
        raise FileNotFoundError(f"NIfTI CSV does not exist: {csv_path}")

    paths: list[Path] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)

        for row_number, row in enumerate(reader, start=1):
            if not row:
                continue

            value = row[0].strip()

            if not value:
                continue

            if row_number == 1 and value.lower() in {
                "nifti",
                "nifti_path",
                "nifti_paths",
                "path",
                "file",
                "filename",
            }:
                continue

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


def _output_name_from_input(input_path: Path) -> str:
    """Generate an individual 5th-percentile map name from the input name."""
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

    return f"{stem}_dfc_dmn_5th.nii.gz"


def dfc_peak_index_null_threshold_map(
    nifti_csv: str | Path,
    dmn_mask_path: str | Path,
    output_dir: str | Path,
) -> DFCResult:
    """Compute control dFC 5th-percentile maps and their voxelwise median.

    Parameters
    ----------
    nifti_csv:
        CSV containing full paths to control 4D NIfTI files in the first
        column. An optional header is allowed.
    dmn_mask_path:
        Full path to the 3D DMN mask NIfTI.
    output_dir:
        Directory where individual 5th-percentile maps and the final
        voxelwise median null-threshold map are written.

    Returns
    -------
    DFCResult
        Paths to the individual 5th-percentile maps and the final
        null-threshold median map.
    """
    nifti_paths = _read_nifti_csv(nifti_csv)

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

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dmn_img, dmn_mask = _load_3d_mask(dmn_mask_path)

    individual_paths: list[str] = []
    all_maps: list[np.ndarray] = []
    reference_shape: tuple[int, int, int] | None = None

    for i, nifti_path in enumerate(nifti_paths, start=1):
        print(
            f"Processing {i}/{len(nifti_paths)}: {nifti_path}"
        )

        out_name = _output_name_from_input(nifti_path)
        out_path = output_dir / out_name

        if out_path.is_file():
            print(
                f"  Output already exists; loading existing map: {out_path}"
            )

            existing_img = load_img(str(out_path))
            existing_map = existing_img.get_fdata(dtype=np.float64)

            if existing_map.ndim != 3:
                raise ValueError(
                    f"Expected existing output to be 3D, got shape "
                    f"{existing_map.shape} for {out_path}"
                )

            if reference_shape is None:
                reference_shape = existing_map.shape
            elif existing_map.shape != reference_shape:
                raise ValueError(
                    f"Spatial shape mismatch: {out_path} has "
                    f"{existing_map.shape}, expected {reference_shape}"
                )

            if existing_map.shape != dmn_mask.shape:
                raise ValueError(
                    f"Existing map shape {existing_map.shape} does not "
                    f"match DMN mask shape {dmn_mask.shape}: {out_path}"
                )

            all_maps.append(existing_map)
            individual_paths.append(str(out_path))
            continue

        ref_img, data = _load_4d_data(nifti_path)

        if data.shape[:3] != dmn_mask.shape:
            raise ValueError(
                f"DMN mask shape {dmn_mask.shape} does not match "
                f"input data shape {data.shape[:3]} for {nifti_path}"
            )

        if reference_shape is None:
            reference_shape = data.shape[:3]
        elif data.shape[:3] != reference_shape:
            raise ValueError(
                f"Spatial shape mismatch: {nifti_path} has "
                f"{data.shape[:3]}, expected {reference_shape}"
            )

        dfc, _ = _compute_dfc_matrix(
            data,
            dmn_mask,
        )

        dfc_5th = np.nanpercentile(
            dfc,
            5,
            axis=1,
        )

        dfc_map = dfc_5th.reshape(data.shape[:3])
        all_maps.append(dfc_map)

        individual_paths.append(
            _save_3d_map(
                ref_img,
                dfc_map,
                out_path,
            )
        )

        print(f"Saved: {out_path}")

    if not all_maps:
        raise ValueError("No individual dFC maps were available for median")

    stack = np.stack(
        all_maps,
        axis=0,
    )

    median_map = np.nanmedian(
        stack,
        axis=0,
    )

    median_out = (
        output_dir
        / "dfc_peak_index_null_threshold_map.nii.gz"
    )

    _save_3d_map(
        dmn_img,
        median_map,
        median_out,
    )

    print(f"Saved null-threshold map: {median_out}")

    return DFCResult(
        individual_map_paths=individual_paths,
        median_map_path=str(median_out),
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--nifti-csv",
        required=True,
        help=(
            "CSV containing full paths to control 4D NIfTI files "
            "in the first column."
        ),
    )

    parser.add_argument(
        "--dmn-mask",
        required=True,
        help="Full path to the 3D DMN mask NIfTI.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Directory for individual 5th-percentile maps and "
            "the final null-threshold map."
        ),
    )

    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    dfc_peak_index_null_threshold_map(
        nifti_csv=args.nifti_csv,
        dmn_mask_path=args.dmn_mask,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "DFCResult",
    "dfc_peak_index_null_threshold_map",
]
