# Retinal Blood Vessel Segmentation

Automated segmentation of retinal blood vessels from fundus images using Frangi vesselness filtering.

## Requirements

- MATLAB
- Image Processing Toolbox

## Usage

Single image:
```matlab
dice_score = retinal_segmentation_pipeline('DRIVE/images/21_training.tif', 'DRIVE/1st_manual/21_manual1.gif', true);
```

Run benchmark on all images:
```matlab
benchmark
```

Results are saved in `/results` as PNG files.

## Method

1. Green channel extraction and preprocessing
2. Frangi vesselness filter (multi-scale)
3. Hysteresis thresholding
4. Morphological cleanup

## Dataset

Uses the DRIVE (Digital Retinal Images for Vessel Extraction) dataset.
