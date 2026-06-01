# dfc-peak-index
Method and analysis code for the manuscript: Sainburg LE, Roche A, Makhoul GS, Rogers BP, Williams Roberson S, Meletti S, Vaudano AE, Chang C, Englot DJ, Morgan VL. The dynamic functional connectivity peak index: detection of interictal epileptic activity with fMRI. *Epilepsia* (accepted).

## Analysis
The script and data to reproduce the main figures and analysis from this manuscript are located at data/dfc_peak_index_variables.csv and code/dfc_peak_index_analysis_plot.m

## dFC Peak Index Method
The main script to compute dFC peak index maps on a group of scans is located at code/dfc_peak_index_dmn.m. This script takes in a csv list of paths to NIFTI fMRI images in MNI space, a csv of the paths to the NIFTI control 5th percentile images (generated from code/dmn_dfc_5th_perc.m; used to set dFC threshold), the mask for the DMN, and the output directory for the images. The script to generate the control 5th percentile maps is located at code/dmn_dfc_5th_perc.m and needs to be run before code/dfc_peak_index_dmn.m.

### Notes
- SPM12 is a dependency of these scripts. Functions from other software packages could be substituted to load in images
- This method and scripts assume consistency of TR and number of timepoints across subjects/scans. Adjustments need to be applied if implemented on datasets with heterogeneous acquisitions
- These scripts assume that all images are in the same space (e.g. MNI152)
- We used a full anatomical parcellation of the DMN as a seed for the dFC peak index method, but other networks, seeds, or parcellations could be used for this method. An example of a single subject's DMN parcellation from the study is at data/dmn.nii.gz.
