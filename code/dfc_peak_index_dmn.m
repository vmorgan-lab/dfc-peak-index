%% This script calculates the number of negative dFC peaks to the DMN for every voxel

% L Sainburg April 2025


function dfc_peak_index_dmn(subject_csv, control_5th_csv, dmn_mask_path, output_dir)
% COMPUTE_DMN_DFC_NEG_PEAKS
% Counts negative dFC peaks to the DMN using a control-derived threshold.
%
% Inputs:
%   subject_csv: CSV with subject fMRI NIFTI paths for processing
%   control_5th_csv: CSV of healthy control DMN DFC 5th percentile maps for
%   null model thresholding (computed with dmn_dfc_5th_perc.m)
%   dmn_mask_path: path to DMN mask NIfTI (can define it with any atlas)
%   output_dir: directory to save outputs

    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    %% Load subject paths
    sub_tbl = readtable(subject_csv, 'ReadVariableNames', false);
    subject_paths = sub_tbl{:,1};
    if isstring(subject_paths)
        subject_paths = cellstr(subject_paths);
    end
    subject_paths = subject_paths(~cellfun('isempty', subject_paths));

    %% Load control 5th percentile maps
    con_tbl = readtable(control_5th_csv, 'ReadVariableNames', false);
    control_paths = con_tbl{:,1};
    if isstring(control_paths)
        control_paths = cellstr(control_paths);
    end
    control_paths = control_paths(~cellfun('isempty', control_paths));

    %% Load DMN mask
    dmn_info = spm_vol(dmn_mask_path);
    dmn = spm_read_vols(dmn_info);
    dims = size(dmn);

    %% --- STEP 1: Load control maps and compute median ---
    fprintf('Loading control 5th percentile maps...\n');

    n_vox = prod(dims);
    n_cons = length(control_paths);

    dmn_dfc_cons = nan(n_vox, n_cons);

    for i = 1:n_cons
        info = spm_vol(control_paths{i});
        vol = spm_read_vols(info);

        dmn_dfc_cons(:, i) = vol(:);
    end

    % Median across controls
    valid_mask = sum(~isnan(dmn_dfc_cons), 2) >= 1; % number of controls with value at this voxel, change if needed
    dmn_dfc_con_med = nanmedian(dmn_dfc_cons, 2);
    dmn_dfc_con_med(~valid_mask) = NaN;

    %% --- STEP 2: Process each subject ---
    fprintf('Computing negative peaks...\n');

    dmn_vx = reshape(dmn, [], 1) > 0;

    for i = 1:length(subject_paths)

        fprintf('Processing %d/%d\n', i, length(subject_paths));

        data_info = spm_vol(subject_paths{i});
        data = spm_read_vols(data_info);

        dims_data = size(data);
        n_vox = prod(dims_data(1:3));
        n_time = dims_data(4);

        data_vx = reshape(data, n_vox, n_time);

        %% DMN time series
        data_dmn = mean(data_vx(dmn_vx, :), 1, 'omitnan');

        %% Normalize
        data_vx_norm = normalize(data_vx, 2);
        data_dmn_norm = normalize(data_dmn, 2);

        %% dFC
        dfc = data_vx_norm .* data_dmn_norm;

        %% Count negative peaks
        dfc_negs = nan(n_vox, 1);

        for v = 1:n_vox
            if ~isnan(dfc(v,1)) && ~isnan(dmn_dfc_con_med(v))

                % Peaks below threshold
                [~, locs1] = findpeaks(-dfc(v,:),'MinPeakHeight', -dmn_dfc_con_med(v));

                % DMN signal must be negative
                locs2 = find(data_dmn < 0);

                dfc_negs(v) = length(intersect(locs1, locs2));
            end
        end

        %% Reshape and save
        dfc_map = reshape(dfc_negs, dims_data(1:3));

        [~, name, ~] = fileparts(subject_paths{i});
        out_path = fullfile(output_dir, [name '_dfc_dmn_negpeaks.nii']);

        out_info = data_info(1);
        out_info.fname = out_path;
        out_info.dt = [64 0];
        out_info.pinfo = [1; 0; 0];

        spm_write_vol(out_info, dfc_map);

        fprintf('Saved: %s\n', out_path);
    end

end