%% This script calculates the 5th percentile of the dFC distribution between
% every voxel and the DMN to use as a threshold for detecting DMN peaks

% L Sainburg April 2025
% SPM12 is a dependency of this script

function dmn_dfc_5th_perc(csv_file, dmn_mask_path, output_dir)
% COMPUTE_DMN_DFC_5TH
% Calculates the 5th percentile of dFC between each voxel and the DMN.
%
% Inputs:
%   csv_file: CSV file with NIfTI paths for the controls you want to include in the null model threshold
%   dmn_mask_path: path to DMN mask NIfTI (can define it with any atlas)
%   output_dir: directory to save output maps

% NOTE that all images (DMN mask, fMRI images) need to be in the same space
% (e.g. MNI152) and resolution.

    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    %% Load NIfTI paths from CSV
    tbl = readtable(csv_file, 'ReadVariableNames', false);
    nifti_paths = tbl{:,1};

    % Ensure cell array of char
    if isstring(nifti_paths)
        nifti_paths = cellstr(nifti_paths);
    end

    % Remove empty rows
    nifti_paths = nifti_paths(~cellfun('isempty', nifti_paths));

    %% Load DMN mask
    dmn_info = spm_vol(dmn_mask_path);
    dmn = spm_read_vols(dmn_info);
    dmn_vx = reshape(dmn, [], 1) > 0;

    %% Loop over subjects
    for i = 1:length(nifti_paths)

        fprintf('Processing %d/%d\n', i, length(nifti_paths));

        % Load subject data
        data_info = spm_vol(nifti_paths{i});
        data = spm_read_vols(data_info);

        dims = size(data);
        n_vox = prod(dims(1:3));
        n_time = dims(4);

        % Reshape to [voxels x time]
        data_vx = reshape(data, n_vox, n_time);

        %% DMN mean time series
        data_dmn = mean(data_vx(dmn_vx, :), 1, 'omitnan');

        %% Normalize
        data_vx_norm = normalize(data_vx, 2);
        data_dmn_norm = normalize(data_dmn, 2);

        %% dFC computation
        dfc_dmn = data_vx_norm .* data_dmn_norm;

        %% 5th percentile across time
        dfc_5th = prctile(dfc_dmn, 5, 2);

        %% Back to 3D
        dfc_map = reshape(dfc_5th, dims(1:3));

        %% Save output
        [~, name, ~] = fileparts(nifti_paths{i});
        out_path = fullfile(output_dir, [name '_dfc_dmn_5th.nii']);

        out_info = data_info(1);
        out_info.fname = out_path;
        out_info.dt = [64 0];
        out_info.pinfo = [1; 0; 0];

        spm_write_vol(out_info, dfc_map);

        fprintf('Saved: %s\n', out_path);
    end
end