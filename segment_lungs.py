import logging
import os 
import numpy as np

from pydicom import dcmread
from lungmask import mask
import SimpleITK as sitk
from tqdm import tqdm

"""
Some volumes are not primary/original (i.e. lung scan 2, 58).
"""

NUM_CT_SCANS = 422
PATH = os.path.join(os.getcwd(), "nsclc_data/NSCLC-Radiomics")

def read_dicoms(path, primary=True, original=True):
    allfnames = []
    
    for dir, _, fnames in os.walk(path):
        [allfnames.append(os.path.join(dir, fname)) for fname in fnames]
    
    dcm_header_info = []
    dcm_parameters = []
    unique_set = []  # need this because too often there are duplicates of dicom files with different names
    for fname in tqdm(allfnames):
        filename_ = os.path.splitext(os.path.split(fname)[1])
        if filename_[0] != 'DICOMDIR':
            try:
                dicom_header = dcmread(fname, defer_size=100, stop_before_pixels=True, force=True)
                if dicom_header is not None:
                    if 'ImageType' in dicom_header:
                        print(dicom_header.ImageType)
                        if primary:
                            print([x in dicom_header.ImageType for x in ['PRIMARY']])
                            is_primary = all([x in dicom_header.ImageType for x in ['PRIMARY']])
                        else:
                            is_primary = True

                        if original:
                            is_original = all([x in dicom_header.ImageType for x in ['ORIGINAL']])
                        else:
                            is_original = True
                            
                        if is_primary and is_original and 'LOCALIZER' not in dicom_header.ImageType:
                            print("IS PRIMARY AND ORIGINAL")
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

    return relevant_volumes


for i in range(1, NUM_CT_SCANS+1):
    if i < 10:
        scan = f"LUNG1-00{i}"
    elif i >= 10 and i < 100: 
        scan = f"LUNG1-0{i}"
    else:
        scan = f"LUNG1-{i}"
    
    # Retrieve folder 
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
    img = mask.utils.read_dicoms(CUR_PATH, primary=False, original=False)[0] # NOTE: image does not have to be primary or original image
        
    assert sitk.GetArrayViewFromImage(img).ndim == 3
    assert np.any(sitk.GetArrayViewFromImage(img))
    
    msk = mask.apply(img)
    
    assert msk.ndim == 3
    assert np.any(msk)
    
    # Save .npy file 
    CUR_PATH = os.path.join(os.getcwd(), f'segmented_nsclc_data/LUNG-{i}')
    np.save(CUR_PATH, msk)
    
    
    