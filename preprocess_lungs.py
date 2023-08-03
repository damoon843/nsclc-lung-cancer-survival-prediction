import numpy as np
import matplotlib.pyplot as plt
import os
import logging

from scipy.ndimage import zoom
import SimpleITK as sitk
from pydicom import dcmread
from lungmask import mask

### CONSTANTS ### 

NUM_CT_SCANS = 422
# For windowing 
WIDTH = 1500
LEVEL = -600
# For resizing 
NEW_DEPTH = 64 
NEW_HEIGHT = 128
NEW_WIDTH = 128

### HELPER FUNCTIONS ### 
def resample_volume(volume, slice_thickness, pixel_spacing):
    """ 
    Helper to resample slice so the size of each voxel is 1x1x1 mm^3.
    Code adapted from: https://www.kaggle.com/code/akh64bit/full-preprocessing-tutorial
    """
    new_spacing = [1, 1, 1]
    
    x_pixel = np.float64(pixel_spacing[0])
    y_pixel = np.float64(pixel_spacing[1])
    original_spacing = np.array([np.float64(slice_thickness), y_pixel, x_pixel]) # (z, y, x)
    
    resize_factor = original_spacing / new_spacing
    new_shape = np.round(volume.shape * resize_factor)
    real_resize_factor = new_shape / volume.shape 

    resampled_volume = zoom(volume, real_resize_factor)
    
    return resampled_volume


def crop_images(images):
    """
    Helper to crop each volume to the smallest non-zero bounding box. 
    NOTE: numpy dimensions are (z, y, x).
    
    Code modified from modified from: https://www.kaggle.com/code/josecarmona/btrc-a-simple-tool-to-crop-3d-images
    """
    _min=np.array(np.nonzero(images)).min(axis=1)
    _max=np.array(np.nonzero(images)).max(axis=1)+1 # --> Thank you @lai321!
    
    return _min, _max
        
        
def resize_images(images):
    """
    Resize volume.
    Code adapted from https://keras.io/examples/vision/3D_image_classification/
    """
    # Get current depth
    current_depth = images.shape[0]
    current_height = images.shape[1]
    current_width = images.shape[2]
    # Compute depth factor
    depth = current_depth / NEW_DEPTH
    width = current_width / NEW_WIDTH
    height = current_height / NEW_HEIGHT
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Resize across z-axis
    resized_images = zoom(images, (depth_factor, height_factor, width_factor), order=1)
    
    assert resized_images.shape[0] == NEW_DEPTH
    assert resized_images.shape[1] == NEW_HEIGHT
    assert resized_images.shape[2] == NEW_WIDTH
    
    return resized_images
    

def preprocess_pixels(volume, slope, intercept, level, width):
    """
    Helper to convert pixels to HU and apply windowing.
    """
    # Convert to HU
    volume = (volume * slope +intercept)
    
    # Windowing
    _min = level - width/2 # Minimum HU 
    _max = level + width/2 # Maximum HU
    volume[volume < _min] = _min 
    volume[volume > _max] = _max 
    
    # Normalize to (0, 1)
    volume = (volume - _min) / (_max - _min)

    return volume

def read_dicoms(path, primary=True, original=True):
    """
    Code modified from https://github.com/JoHof/lungmask
    Include logic that returns rescale slope, rescale intercept, pixel spacing, and slice thickness.
    """ 
    allfnames = []
    for dir, _, fnames in os.walk(path):
        [allfnames.append(os.path.join(dir, fname)) for fname in fnames]
    
    allfnames.sort(key=lambda file: int(file.split("/")[-1].split("-")[-1].split(".")[0])) # Sort slices
    
    dcm_header_info = []
    dcm_parameters = []
        
    unique_set = []  # need this because too often there are duplicates of dicom files with different names
    for i,fname in enumerate(allfnames):
        filename_ = os.path.splitext(os.path.split(fname)[1])
        if filename_[0] != 'DICOMDIR':
            try:
                dicom_header = dcmread(fname, defer_size=100, stop_before_pixels=True, force=True)
                if dicom_header is not None:
                    
                    assert dicom_header.PixelSpacing is not None 
                    assert dicom_header.ImagePositionPatient is not None
                    assert dicom_header.RescaleSlope is not None 
                    assert dicom_header.RescaleIntercept is not None
                    
                    pixel_spacing = dicom_header.PixelSpacing                    
                    if i == 0:
                        slope = dicom_header.RescaleSlope
                        intercept = dicom_header.RescaleIntercept
                        first_thickness = dicom_header.ImagePositionPatient[2] # Position of Z axis
                        
                    if i == 1:
                        second_thickness = dicom_header.ImagePositionPatient[2]
                        slice_thickness = np.abs(second_thickness - first_thickness)                        
                                                
                    if 'ImageType' in dicom_header:                        
                        if primary:
                            is_primary = all([x in dicom_header.ImageType for x in ['PRIMARY']])
                        else:
                            is_primary = True

                        if original:
                            is_original = all([x in dicom_header.ImageType for x in ['ORIGINAL']])
                        else:
                            is_original = True
                            
                        if is_primary and is_original and 'LOCALIZER' not in dicom_header.ImageType:
                            h_info_wo_name = [dicom_header.StudyInstanceUID, dicom_header.SeriesInstanceUID,
                                              dicom_header.ImagePositionPatient]
                            h_info = [dicom_header.StudyInstanceUID, dicom_header.SeriesInstanceUID, fname,
                                      dicom_header.ImagePositionPatient]
                            if h_info_wo_name not in unique_set:
                                unique_set.append(h_info_wo_name)
                                dcm_header_info.append(h_info)
            
            except:
                logging.error("Unexpected error:", sys.exc_info()[0])
                logging.warning("Doesn't seem to be DICOM, will be skipped: ", fname)
            
    conc = [x[1] for x in dcm_header_info]
    sidx = np.argsort(conc)
    conc = np.asarray(conc)[sidx]
    dcm_header_info = np.asarray(dcm_header_info, dtype=object)[sidx]
    # dcm_parameters = np.asarray(dcm_parameters)[sidx]
    vol_unique = np.unique(conc, return_index=1, return_inverse=1)  # unique volumes
    n_vol = len(vol_unique[1])
    if n_vol == 1:
        logging.info('There is ' + str(n_vol) + ' volume in the study')
    else:
        logging.info('There are ' + str(n_vol) + ' volumes in the study')

    relevant_series = []
    relevant_volumes = []

    for i in range(len(vol_unique[1])):
        curr_vol = i
        info_idxs = np.where(vol_unique[2] == curr_vol)[0]
        vol_files = dcm_header_info[info_idxs, 2]
        positions = np.asarray([np.asarray(x[2]) for x in dcm_header_info[info_idxs, 3]])
        slicesort_idx = np.argsort(positions)
        vol_files = vol_files[slicesort_idx]
        relevant_series.append(vol_files)
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(vol_files)
        vol = reader.Execute()
        relevant_volumes.append(vol)

    return relevant_volumes, slope.real, intercept.real, pixel_spacing, slice_thickness


### PREPROCESS CODE ### 

lungs = []
skipped = [] # Store scans that are not ORIGINAL

for i in range(1, NUM_CT_SCANS+1):
    # Load original lung scan
    print(f"SCAN {i}")
    PATH = os.path.join(os.getcwd(), "nsclc_data/NSCLC-Radiomics")
    if i < 10:
        scan = f"LUNG1-00{i}"
    elif i >= 10 and i < 100: 
        scan = f"LUNG1-0{i}"
    else:
        scan = f"LUNG1-{i}"

    CUR_PATH = os.path.join(PATH, scan)
    res = [file for file in os.listdir(CUR_PATH) if not file.startswith(".")][0]
    # Retrieve and sort folders -- full lung CT scan will start with lowest number
    CUR_PATH = os.path.join(CUR_PATH, res)
    res = []
    for file in os.listdir(CUR_PATH):
        if not file.startswith(".") and "segmentation" not in file.lower():
            res.append(file) 
    idx = np.argsort([int(file.split(".")[0]) for file in res])
    # Read in DICOMs and apply mask
    CUR_PATH = os.path.join(CUR_PATH, res[idx[0]])
    img, slope, intercept, pixel_spacing, slice_thickness = read_dicoms(CUR_PATH, primary=False, original=True)
    if len(img) == 0:
        skipped.append(i)
        print(f"SKIPPING SCAN{i}")
        continue
    
    volume = img[0]    
    img_arr = sitk.GetArrayFromImage(volume)
    # Load and apply mask
    _mask = np.load(os.path.join(os.getcwd(), f'segmented_nsclc_data/LUNG-{i}.npy'))
    _mask[_mask == 2] = 1
    masked_lung = _mask * img_arr    
    # Crop to smallest bounding box
    _min, _max = crop_images(masked_lung)
    cropped_lung = masked_lung[_min[0]:_max[0], _min[1]: _max[1], _min[2]: _max[2]]
    # Convert to HU
    preprocessed_lung = preprocess_pixels(cropped_lung, slope, intercept, LEVEL, WIDTH)    
    # Resize to (64, 128, 128)
    resized_lung = resize_images(preprocessed_lung)
    lungs.append(resized_lung)  

assert len(lungs) == 421

np.save(os.path.join(os.getcwd(), 'preprocessed_lungs.npy'),  lungs, allow_pickle=True)

print(f"Skipped scans: {skipped}")
    

