"""Compute patient DMN-coupled peak-frequency maps.

This module can be used either as a standalone CLI or imported directly from
``postprocessing.py``. It searches BIDS derivatives/fmri_postproc for smoothed
postprocessed NIfTI files and writes one peak-frequency map per input run into
the matching session-level dfc_peak_index directory.

Thresholding options
--------------------
You can provide either:
1. a 3D null-threshold map (voxelwise thresholding), or
2. a constant threshold value (applied to all voxels).

Timing
------
TR is read from each input NIfTI header.

Supported input layout
----------------------
BIDS/derivatives/fmri_postproc/sub-*/ses*/nifti/
    sub-*_ses-*_task-rest*_space-MNI152NLin2009cAsym_res-2_desc-postprocessed_smooth-4mm_bold.nii.gz

Output layout
-------------
BIDS/derivatives/fmri_postproc/sub-*/ses*/dfc_peak_index/
    same filename as the input NIfTI, but with
    desc-postprocessed_dfc-peak-index_smooth_4mm.nii.gz
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import load_img, new_img_like
from scipy.signal import find_peaks


def _normalize_subject(value: str | int) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("Encountered an empty subject value")
    return text if text.startswith("sub-") else f"sub-{text}"


def _normalize_subject_list(values: Iterable[str | int]) -> list[str]:
    return [_normalize_subject(v) for v in values if str(v).strip()]


def _read_subject_csv(csv_file: str | Path) -> list[str]:
    df = pd.read_csv(csv_file, header=None)
    if df.shape[1] < 1:
        raise ValueError(f"Expected a CSV with a subject column: {csv_file}")

    first_value = str(df.iat[0, 0]).strip().lower()
    if first_value in {"subject", "sub", "participant", "participant_id"}:
        df = df.iloc[1:].reset_index(drop=True)

    subjects: list[str] = []
    for value in df.iloc[:, 0].tolist():
        if pd.isna(value):
            continue
        subjects.append(_normalize_subject(value))

    if not subjects:
        raise ValueError(f"No valid subject names found in CSV: {csv_file}")
    return subjects


def _load_4d_data(nifti_path: str | Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    img = load_img(str(nifti_path))
    data = img.get_fdata(dtype=np.float64)
    if data.ndim != 4:
        raise ValueError(f"Expected a 4D fMRI image, got shape {data.shape} for {nifti_path}")
    return img, data


def _load_3d_mask(mask_path: str | Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    img = load_img(str(mask_path))
    data = img.get_fdata(dtype=np.float64)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape {data.shape} for {mask_path}")
    return img, data > 0


def _get_tr_from_img(img: nib.Nifti1Image, nifti_path: str | Path) -> float:
    zooms = img.header.get_zooms()
    if len(zooms) < 4 or not np.isfinite(zooms[3]) or zooms[3] <= 0:
        raise ValueError(f"Could not read a valid TR from NIfTI header for {nifti_path}")
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
        out[valid_rows] = (x[valid_rows] - mean[valid_rows]) / std[valid_rows]
        out[~good] = np.nan
    if np.any(~valid_rows):
        out[~valid_rows] = 0.0
        out[~good & ~valid_rows[:, None]] = np.nan
    return out


def _compute_dfc_matrix(data_4d: np.ndarray, dmn_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if data_4d.ndim != 4:
        raise ValueError("data_4d must be 4D")
    if dmn_mask.shape != data_4d.shape[:3]:
        raise ValueError(
            f"DMN mask shape {dmn_mask.shape} does not match data spatial shape {data_4d.shape[:3]}"
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


def _save_3d_map(reference_img: nib.Nifti1Image, data_3d: np.ndarray, out_path: str | Path) -> str:
    out_path = str(out_path)
    img = new_img_like(reference_img, data_3d.astype(np.float32), copy_header=True)
    nib.save(img, out_path)
    return out_path


def _find_subject_sessions(bids_root: str | Path, subject: str) -> list[str]:
    bids_root = Path(bids_root)
    subject_dir = bids_root / "derivatives" / "fmri_postproc" / subject
    if not subject_dir.exists():
        raise FileNotFoundError(f"Missing subject directory: {subject_dir}")

    sessions = sorted(p.name for p in subject_dir.glob("ses-*") if p.is_dir())
    if not sessions:
        raise FileNotFoundError(f"No session directories found under: {subject_dir}")
    return sessions


def _find_bids_session_niftis(bids_root: str | Path, subject: str, session: str) -> list[Path]:
    bids_root = Path(bids_root)
    nifti_dir = bids_root / "derivatives" / "fmri_postproc" / subject / session / "nifti"
    if not nifti_dir.exists():
        raise FileNotFoundError(f"Missing nifti directory: {nifti_dir}")

    pattern = (
        f"{subject}_{session}_task-rest*_"
        "space-MNI152NLin2009cAsym_res-2_desc-postprocessed_smooth-4mm_bold.nii.gz"
    )
    paths = sorted(nifti_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No matching NIfTI files found in: {nifti_dir}")
    return paths


def _output_peak_filename(
    input_path: Path,
    dfc_seed_name: str | None = None,
) -> str:
    seed_tag = f"_{dfc_seed_name}" if dfc_seed_name else ""
    name = input_path.name
    suffix = "_desc-postprocessed_smooth-4mm_bold.nii.gz"
    replacement = (
        f"_desc-postprocessed_dfc-peak-index"
        f"{seed_tag}_smooth_4mm.nii.gz"
    )
    if name.endswith(suffix):
        return name[:-len(suffix)] + replacement
    if name.endswith(".nii.gz"):
        stem = name[:-7]
        return (
            re.sub(
                r"_desc-[^_]+_smooth-4mm_bold$",
                f"_desc-postprocessed_dfc-peak-index{seed_tag}_smooth_4mm",
                stem,
            )
            + ".nii.gz"
        )
    raise ValueError(f"Unexpected input filename format: {name}")


def _find_control_negative_peaks(ts: np.ndarray, thr_v: float) -> np.ndarray:
    # Peaks in -dFC are local minima in dFC.
    peaks, _ = find_peaks(-ts, height=-thr_v)
    return peaks


def _prepare_thresholds(
    threshold_map_path: str | Path | None,
    threshold_value: float | None,
) -> tuple[np.ndarray | float, str]:
    if (threshold_map_path is None) == (threshold_value is None):
        raise ValueError("Provide exactly one of threshold_map_path or threshold_value")

    if threshold_map_path is not None:
        thr_img = load_img(str(threshold_map_path))
        thr = thr_img.get_fdata(dtype=np.float64)
        if thr.ndim != 3:
            raise ValueError(f"Expected a 3D threshold map, got shape {thr.shape}")
        return thr, "map"

    assert threshold_value is not None
    if not np.isfinite(threshold_value):
        raise ValueError("threshold_value must be finite")
    return float(threshold_value), "constant"


def compute_dfc_peak_index(
    bids_root: str | Path,
    dmn_mask_path: str | Path,
    threshold_map_path: str | Path | None = None,
    threshold_value: float | None = None,
    subject_csv: str | Path | None = None,
    subject_ids: list[str] | tuple[str, ...] | str | None = None,
    dfc_seed_name: str | None = None,
) -> list[str]:
    """Compute DMN-coupled peak-frequency maps for every run in every subject session.

    Parameters
    ----------
    bids_root:
        BIDS dataset root containing derivatives/fmri_postproc.
    dmn_mask_path:
        3D DMN mask NIfTI path.
    threshold_map_path:
        Optional 3D voxelwise null-threshold map derived from control
        5th-percentile dFC maps.
    threshold_value:
        Optional constant threshold applied to all voxels.
    subject_csv:
        Optional CSV containing a single subject column.
    subject_ids:
        Optional subject ID or list of subject IDs to process directly.

    Returns
    -------
    list[str]
        Paths to the peak-frequency maps.
    """
    bids_root = Path(bids_root)
    threshold_field, threshold_mode = _prepare_thresholds(threshold_map_path, threshold_value)
    _, dmn_mask = _load_3d_mask(dmn_mask_path)

    if subject_ids is not None:
        if isinstance(subject_ids, (str, Path)):
            subjects = [_normalize_subject(subject_ids)]
        else:
            subjects = _normalize_subject_list(subject_ids)
    elif subject_csv is not None:
        subjects = _read_subject_csv(subject_csv)
    else:
        raise ValueError("Provide subject_csv or subject_ids")

    out_paths: list[str] = []

    for subject in subjects:
        sessions = _find_subject_sessions(bids_root, subject)
        for session in sessions:
            session_paths = _find_bids_session_niftis(bids_root, subject, session)
            output_dir = bids_root / "derivatives" / "fmri_postproc" / subject / session / "dfc_peak_index"
            output_dir.mkdir(parents=True, exist_ok=True)

            for i, nifti_path in enumerate(session_paths, start=1):
                print(f"Processing {subject} {session} {i}/{len(session_paths)}: {nifti_path}")
                img, data = _load_4d_data(nifti_path)
                tr = _get_tr_from_img(img, nifti_path)

                if data.shape[:3] != dmn_mask.shape:
                    raise ValueError(
                        f"DMN mask shape {dmn_mask.shape} does not match subject data shape {data.shape[:3]}"
                    )

                if threshold_mode == "map":
                    thr = np.asarray(threshold_field, dtype=np.float64)
                    if thr.shape != data.shape[:3]:
                        raise ValueError(
                            f"Threshold map shape {thr.shape} does not match subject data shape {data.shape[:3]}"
                        )
                    thr_flat = thr.reshape(-1)
                else:
                    thr_flat = float(threshold_field)

                dfc, dmn_ts = _compute_dfc_matrix(data, dmn_mask)
                n_vox, n_time = dfc.shape
                dmn_negative = np.where(dmn_ts < 0)[0]

                peak_freq = np.full(n_vox, np.nan, dtype=np.float64)
                for v in range(n_vox):
                    ts = dfc[v, :]
                    if not np.all(np.isfinite(ts)):
                        continue

                    thr_v = thr_flat[v] if threshold_mode == "map" else thr_flat
                    if not np.isfinite(thr_v):
                        continue

                    peaks = _find_control_negative_peaks(ts, float(thr_v))
                    if peaks.size == 0 or dmn_negative.size == 0:
                        peak_freq[v] = 0.0
                        continue

                    count = np.intersect1d(peaks, dmn_negative, assume_unique=False).size
                    peak_freq[v] = count / (tr * n_time)

                out_map = peak_freq.reshape(data.shape[:3])
                out_name = _output_peak_filename(nifti_path,dfc_seed_name=dfc_seed_name)
                out_path = output_dir / out_name
                out_paths.append(_save_3d_map(img, out_map, out_path))
                print(f"Saved: {out_path}")

    return out_paths


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids-root", required=True, help="BIDS dataset root")
    parser.add_argument("--subject-csv", required=True, help="CSV with a subject column")
    parser.add_argument("--dmn-mask", required=True, help="3D DMN mask NIfTI")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--null-threshold-map",
        dest="threshold_map_path",
        help="3D null-threshold map (voxelwise thresholding)",
    )
    group.add_argument(
        "--threshold-value",
        dest="threshold_value",
        type=float,
        help="Constant threshold applied to all voxels",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    compute_dfc_peak_index(
        bids_root=args.bids_root,
        subject_csv=args.subject_csv,
        dmn_mask_path=args.dmn_mask,
        threshold_map_path=args.threshold_map_path,
        threshold_value=args.threshold_value,
    )


if __name__ == "__main__":
    main()


__all__ = ["compute_dfc_peak_index"]
