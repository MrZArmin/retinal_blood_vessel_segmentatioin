function dice_score = retinal_segmentation_pipeline(img_path, mask_path, visualize)
    %% 1. Load training image and its manual mask
    img = imread(img_path); 
    manual_mask = logical(imread(mask_path));
   
    %% 2. Preprocess image

    % Only keep the green channel
    green = img(:,:,2); 
    
    % Create a mask to ignore the black camera border so it is not detected
    % as a vessel
    fov_mask = green > 30;
    fov_mask = imerode(fov_mask, strel('disk', 5));
    
    % Invert the image
    green_inv = imcomplement(green);
    % We use a gentle CLAHE to avoid boosting background noise
    preprocessed = adapthisteq(green_inv, 'NumTiles', [8 8], 'ClipLimit', 0.01);
    
    %% 3. Frangi Vesselness Filter
    % We scan 1 to 3 to catch the main trunks too
    scales = [1, 2, 3]; 
    vesselness = zeros(size(preprocessed));
    
    for sigma = scales
        Iv = FrangiFilter2D(preprocessed, sigma);
        vesselness = max(vesselness, Iv);
    end
    
    % Zero out everything outside the retina using the mask
    vesselness = vesselness .* double(fov_mask);

    %% 4. Post-processing: HYSTERESIS THRESHOLDING
    t_low = 0.06;   % Very sensitive (catches thin capillaries)
    t_high = 0.12;  % Strict (catches main trunks)
    
    marker = vesselness > t_high;
    mask = vesselness > t_low;
    
    % "Reconstruct" keeps faint 'mask' pixels ONLY if they connect to 'marker'
    binary_mask = imreconstruct(marker, mask);
    
    % Clean up tiny detached specks
    binary_mask = bwareaopen(binary_mask, 50);

    %% 5. Evaluation
    dice_score = dice(binary_mask, manual_mask);
    fprintf('Final Dice Score: %.4f\n', dice_score);

    %% 6. Comprehensive Visualization
    if visualize
        figure('Name', 'Pipeline Steps', 'NumberTitle', 'off', 'Color', 'w');
        
        subplot(2, 3, 1); 
        imshow(img); 
        title('1. Original Image', 'Color', 'k');
        
        subplot(2, 3, 2); 
        imshow(manual_mask); 
        title('2. Manual Ground Truth', 'Color', 'k');
    
        subplot(2, 3, 3)
        imshow(preprocessed)
        title('3. Preprocessed Image (with CLAHE)', 'Color', 'k')
    
        subplot(2, 3, 4); 
        imshow(vesselness, []); 
        colormap(gca, 'jet');
        title('4. Frangi Vesselness (Heatmap)', 'Color', 'k');
        
        subplot(2, 3, 5); 
        imshow(binary_mask); 
        title('5. Final Binary Segmentation', 'Color', 'k');
        
        subplot(2, 3, 6); 
        imshow(labeloverlay(img, binary_mask, 'Transparency', 0.5)); 
        title(['6. Overlay (Dice: ' num2str(dice_score, '%.3f') ')'], 'Color', 'k');
    end

    % save the binary segmentation to /results with its input number
    numStr = regexp(img_path, '\d+', 'match');  
    newFile = [numStr{1} '.png'];
    imwrite(binary_mask, fullfile('results', newFile));
end

%% The FrangiFilter function
function V = FrangiFilter2D(I, sigma)
    I = double(I);
    siz = round(3*sigma);
    [x,y] = meshgrid(-siz:siz, -siz:siz);
    
    % Gaussian
    G = exp(-(x.^2 + y.^2)/(2*sigma^2)); % gaussian smoothing
    G = G / sum(G(:));
    
    % Derivatives
    Gxx = (x.^2 - sigma^2) ./ (sigma^4) .* G; % left to right
    Gyy = (y.^2 - sigma^2) ./ (sigma^4) .* G; % top to bottom
    Gxy = (x.*y) ./ (sigma^4) .* G;          % diagonal
    
    Dxx = imfilter(I, Gxx, 'replicate');
    Dyy = imfilter(I, Gyy, 'replicate');
    Dxy = imfilter(I, Gxy, 'replicate');
    
    % Eigenvalues
    tmp = sqrt((Dxx - Dyy).^2 + 4*Dxy.^2);
    lambda1 = (Dxx + Dyy + tmp) / 2; % along the vessel (should be small)
    lambda2 = (Dxx + Dyy - tmp) / 2; % across the vessel (should be large negative)
    
    % Sort by magnitude
    mu1 = lambda1; mu2 = lambda2;
    check = abs(lambda1) > abs(lambda2);
    mu1(check) = lambda2(check);
    mu2(check) = lambda1(check);
    lambda1 = mu1; lambda2 = mu2;
    
    beta = 0.5;
    c = 15;
    
    % S is the norm of the Hessian, if low means no structure
    % Rb measures the cone vs tube-likeness
    Rb = (lambda1 ./ (lambda2 + eps)).^2;
    S = lambda1.^2 + lambda2.^2;
    
    % scale from 0 to 1 based on how vessel-like the pixel is
    V = exp(-Rb / (2*beta^2)) .* (1 - exp(-S / (2*c^2)));
    V(lambda2 > 0) = 0;
    V = (V - min(V(:))) / (max(V(:)) - min(V(:)));
end