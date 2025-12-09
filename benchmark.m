img_folder = 'DRIVE/images';
mask_folder = 'DRIVE/1st_manual';

files = dir(fullfile(img_folder, '*.tif'));

scores = [];

for k = 1:length(files)
    base_name = files(k).name;
    img_path = fullfile(img_folder, base_name);
    
    % Construct mask filename
    mask_name = strrep(base_name, '_training.tif', '_manual1.gif');
    mask_path = fullfile(mask_folder, mask_name);
    
    % Check if mask exists
    if ~isfile(mask_path)
        warning('Mask not found for %s (Expected: %s)', base_name, mask_name);
        continue;
    end
    
    % Run Segmentation
    try
        d = retinal_segmentation_pipeline(img_path, mask_path, false);
        scores(end+1) = d;
        fprintf('[%d/%d] %s -> Dice: %.4f\n', k, length(files), base_name, d);
    catch ME
        fprintf('Error processing %s: %s\n', base_name, ME.message);
    end
end

fprintf('------------------------------------------------\n');
fprintf('Processing Complete.\n');
fprintf('Average Dice Score: %.4f\n', mean(scores));