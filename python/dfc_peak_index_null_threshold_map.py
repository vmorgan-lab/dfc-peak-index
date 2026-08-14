# This script computes null threshold maps for dFC Peak Index based on the manuscript by Sainburg et al., 2026: 
# The dynamic functional peak index: detection of interictal epileptic activity with fMRI

# L Sainburg May 2026
# Adapted from scripts at https://github.com/vmorgan-lab/dfc-peak-index/

"""Build a BIDS-aware null-threshold map from control DMN dFC 5th-percentile maps.

This preprocessing-stage script is BIDS-aware. It reads a CSV containing
subject/session pairs, finds the matching preprocessed BOLD runs inside a
BIDS derivatives tree, computes a 5th-percentile DMN-coupled dFC map for
one of two modes, and then writes a voxelwise median null-threshold map
across all selected control maps.

Supported input layout
----------------------
BIDS/derivatives/fmri_postproc/sub-*/ses*/nifti/
    sub-*_ses-*_task-rest_*_space-MNI152NLin2009cAsym_res-2_desc-postprocessed_smooth-4mm_bold.nii.gz

Modes
-----
1. Concatenate all runs within each session and compute the 5th percentile
   over the concatenated time series.
2. Select a specific run number within each session and compute the 5th
   percentile from that run only.

Outputs
-------
All subject/session intermediate maps are written into a user-specified flat
output directory. A voxelwise median null-threshold map is also written there.
"""

# Usage example:
# python dfc_peak_index_null_threshold_map.py --bids-root /space/mcdonald-syn01/1/BIDS/VU_dataset --subject-session-csv /space/mcdonald-syn01/1/projects/lucas/rsfmri-github/local/dfc_controls_vu.csv --dmn-mask /space/mcdonald-syn01/1/projects/lucas/rsfmri-github/atlas/tpl-MNI152NLin2009cAsym_res-02_atlas-Schaefer2018_desc-400Parcels17Networks_DMNmask.nii.gz --output-dir /space/mcdonald-syn01/1/BIDS/VU_dataset/derivatives/fmri_postproc/dfc_peak_index --concat-runs

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import load_img, new_img_like

RunMode = Literal["concat", "run"]


@dataclass(frozen=True)
class DFCResult:
    """Paths to the generated intermediate subject/session and group-level outputs."""

    individual_map_paths: list[str]
    median_map_path: str | None = None


def _normalize_subject(value: str | int) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("Encountered an empty subject value in the CSV")
    return text if text.startswith("sub-") else f"sub-{text}"


def _normalize_session(value: str | int) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("Encountered an empty session value in the CSV")
    return text if text.startswith("ses-") else f"ses-{text}"


def _read_subject_session_csv(csv_file: str | Path) -> list[tuple[str, str]]:
    df = pd.read_csv(csv_file, header=None)
    if df.shape[1] < 2:
        raise ValueError(
            f"Expected a CSV with at least two columns (subject, session): {csv_file}"
        )

    first_row = [str(df.iat[0, 0]).strip().lower(), str(df.iat[0, 1]).strip().lower()]
    if first_row[0] in {"subject", "sub", "participant", "participant_id"} and first_row[1] in {"session", "ses"}:
        df = df.iloc[1:].reset_index(drop=True)

    pairs: list[tuple[str, str]] = []
    for _, row in df.iterrows():
        subj = row.iloc[0]
        sess = row.iloc[1]
        if pd.isna(subj) or pd.isna(sess):
            continue
        pairs.append((_normalize_subject(subj), _normalize_session(sess)))

    if not pairs:
        raise ValueError(f"No valid subject/session pairs found in CSV: {csv_file}")
    return pairs


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


def _find_bids_session_niftis(bids_root: str | Path, subject: str, session: str) -> list[Path]:
    bids_root = Path(bids_root)
    nifti_dir = bids_root / "derivatives" / "fmri_postproc" / subject / session / "nifti"
    if not nifti_dir.exists():
        raise FileNotFoundError(f"Missing nifti directory: {nifti_dir}")

    pattern = (
        f"{subject}_{session}_task-rest_*_"
        "space-MNI152NLin2009cAsym_res-2_desc-postprocessed_smooth-4mm_bold.nii.gz"
    )
    paths = sorted(nifti_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No matching NIfTI files found in: {nifti_dir}")
    return paths


def _extract_run_number(path: str | Path) -> int | None:
    match = re.search(r"_run-(\d+)_", Path(path).name)
    if not match:
        return None
    return int(match.group(1))


def _concat_images(paths: list[Path]) -> tuple[nib.Nifti1Image, np.ndarray]:
    first_img, first_data = _load_4d_data(paths[0])
    pieces = [first_data]
    for path in paths[1:]:
        img, data = _load_4d_data(path)
        if img.shape[:3] != first_img.shape[:3]:
            raise ValueError(f"Spatial mismatch for {path}: {img.shape[:3]} != {first_img.shape[:3]}")
        pieces.append(data)
    concat_data = np.concatenate(pieces, axis=3)
    return first_img, concat_data


def _select_session_input(
    session_paths: list[Path],
    run_mode: RunMode,
    run_number: int | None,
) -> tuple[nib.Nifti1Image, np.ndarray, Path]:
    if run_mode == "concat":
        ref_img, data = _concat_images(session_paths)
        return ref_img, data, session_paths[0]

    if run_number is None:
        raise ValueError("run_number must be provided when run_mode='run'")

    for path in session_paths:
        if _extract_run_number(path) == run_number:
            img, data = _load_4d_data(path)
            return img, data, path

    session_name = session_paths[0].parent.parent.name
    raise FileNotFoundError(f"Could not find run-{run_number} in session {session_name}")


def _output_name_from_reference(reference_path: Path, run_mode: RunMode) -> str:
    stem = reference_path.name
    if stem.endswith(".nii.gz"):
        stem = stem[:-7]
    suffix = "concat" if run_mode == "concat" else f"run-{_extract_run_number(reference_path) or 'unknown'}"
    return f"{stem}_dfc_dmn_5th_{suffix}.nii.gz"


def dfc_peak_index_null_threshold_map(
    bids_root: str | Path,
    subject_session_csv: str | Path,
    dmn_mask_path: str | Path,
    output_dir: str | Path,
    run_mode: RunMode = "concat",
    run_number: int | None = None,
) -> DFCResult:
    """Compute control DMN dFC 5th-percentile maps and their voxelwise null-threshold median map.

    Parameters
    ----------
    bids_root:
        BIDS dataset root containing derivatives/fmri_postproc.
    subject_session_csv:
        CSV with subject and session columns.
    dmn_mask_path:
        3D DMN mask NIfTI path.
    output_dir:
        Flat directory where all intermediate maps are written.
    run_mode:
        Either "concat" (concatenate all runs within each session) or "run"
        (use one specific run number within each session).
    run_number:
        Required when run_mode="run".

    Returns
    -------
    DFCResult
        Paths to the intermediate 5th-percentile maps and the null-threshold median map.
    """
    if run_mode not in {"concat", "run"}:
        raise ValueError("run_mode must be either 'concat' or 'run'")
    if run_mode == "run" and run_number is None:
        raise ValueError("run_number is required when run_mode='run'")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = _read_subject_session_csv(subject_session_csv)
    dmn_img, dmn_mask = _load_3d_mask(dmn_mask_path)

    individual_paths: list[str] = []
    all_maps: list[np.ndarray] = []
    reference_shape: tuple[int, int, int] | None = None

    for i, (subject, session) in enumerate(pairs, start=1):
        print(f"Processing {i}/{len(pairs)}: {subject} {session}")
        
        # skip if subject image already exists in output
        pattern = f"{subject}*{session}*dfc_dmn_5th*.nii.gz"
        matching_files = list(output_dir.glob(pattern))
        if matching_files:
            print(f"  Skipping {subject} {session} (output already exists: {matching_files[0]})")
            individual_paths.append(str(matching_files[0]))
            continue

        session_paths = _find_bids_session_niftis(bids_root, subject, session)
        ref_img, data, reference_path = _select_session_input(session_paths, run_mode, run_number)

        if reference_shape is None:
            reference_shape = data.shape[:3]
        elif data.shape[:3] != reference_shape:
            raise ValueError(
                f"Spatial shape mismatch: {reference_path} has {data.shape[:3]}, expected {reference_shape}"
            )

        dfc, _ = _compute_dfc_matrix(data, dmn_mask)
        dfc_5th = np.nanpercentile(dfc, 5, axis=1)
        dfc_map = dfc_5th.reshape(data.shape[:3])
        all_maps.append(dfc_map)

        out_name = _output_name_from_reference(reference_path, run_mode)
        out_path = output_dir / out_name
        individual_paths.append(_save_3d_map(ref_img, dfc_map, out_path))
        print(f"Saved: {out_path}")

    stack = np.stack(all_maps, axis=0)
    median_map = np.nanmedian(stack, axis=0)
    median_out = output_dir / "dfc_peak_index_null_threshold_map.nii.gz"
    _save_3d_map(dmn_img, median_map, median_out)
    print(f"Saved null-threshold map: {median_out}")

    return DFCResult(individual_map_paths=individual_paths, median_map_path=str(median_out))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids-root", required=True, help="BIDS dataset root")
    parser.add_argument("--subject-session-csv", required=True, help="CSV with subject and session columns")
    parser.add_argument("--dmn-mask", required=True, help="3D DMN mask NIfTI")
    parser.add_argument("--output-dir", required=True, help="Flat output directory for intermediate maps")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--concat-runs",
        action="store_true",
        help="Concatenate all runs within each session before computing the 5th percentile",
    )
    mode.add_argument(
        "--run-number",
        type=int,
        help="Use only this run number from each session",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    run_mode: RunMode = "concat" if args.concat_runs else "run"
    dfc_peak_index_null_threshold_map(
        bids_root=args.bids_root,
        subject_session_csv=args.subject_session_csv,
        dmn_mask_path=args.dmn_mask,
        output_dir=args.output_dir,
        run_mode=run_mode,
        run_number=args.run_number,
    )


if __name__ == "__main__":
    main()


__all__ = ["DFCResult", "dfc_peak_index_null_threshold_map"]
