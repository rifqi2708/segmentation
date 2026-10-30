import SimpleITK as sitk
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def get_properties (img):

    img_data = {
        "img_dimension" : img.GetDimension(),
        "img_size" : img.GetSize(),
        "img_orign" : img.GetOrigin(),
        "img_spacing" : img.GetSpacing(),
        "img_direction" : img.GetDirection(),
    }

    return img_data


def validate_mask (img_dir: str, mask_dir: str):
    image_path = Path(img_dir)
    mask_path = Path(mask_dir)

    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)

    mask_array = sitk.GetArrayViewFromImage(mask) #get python array of the mask
    mask_value = np.unique(mask_array) #get unique values of the mask

    #check if the mask is binary
    is_binary = np.all((mask_value == 1) | (mask_value == 0)) 
    if not is_binary: 
        raise RuntimeError(
            f"this mask is not binary: {image_path.name} --> {mask_path.name}\n"
            f"image: {image_path}\n"
            f"mask: {mask_path.name}\n"
            f"pixel values in the mask: {mask_value}"
        )

    #check if the mask is empty
    has_volume = np.any((mask_value == 1))
    if not has_volume:
        raise RuntimeError(
                    f"this mask is empty: {image_path.name} --> {mask_path.name}\n"
                    f"image: {image_path}\n"
                    f"mask: {mask_path.name}\n"
        )

    #get mask and image properties
    image_properties = get_properties(image)
    mask_properties = get_properties(mask)

    #compare image and mask properties
    if image_properties != mask_properties:
        image_text = "\n".join(
                f"  {key}: {value}" for key, value in image_properties.items()
            )
        mask_text = "\n".join(
                f"  {key}: {value}" for key, value in mask_properties.items()
            )
        
        raise RuntimeError(
            f"Mismatch properties for image: {image_path.name}\n"
            f"Maks: {mask_path.name}\n"
            f"\nImage properties:\n{image_text}\n"
            f"\nMask properties:\n{mask_text}"
        )
    return True


#obtaining all mask dirs in output_folder
def mask_technical_qa  (output_folder: str):
    output_dir = Path(output_folder)
    mask_dirs = {}
    
    for subject_dir in sorted(output_dir.iterdir()): #iterate through all subjects
        if subject_dir.is_dir():   
            subject_name = subject_dir.name
            
            for session_dir in sorted(subject_dir.iterdir()): #iterate thorugh all sessions
                img_name = session_dir.name + "_CT-AC.nii.gz"
                img_dir = output_dir.parent / "input" / subject_name / img_name
                mask_dirs = sorted(session_dir.glob("*.nii.gz"))
                
                if not mask_dirs:
                    raise RuntimeError(f"No masks found in {session_dir}")
                
                for mask_dir in mask_dirs: #iterate through all masks
                    status = validate_mask(
                        img_dir,
                        mask_dir
                    )
                
                print (f"All mask for {img_dir} passed the technical QA")


mask_technical_qa ("/workspace/segmentation/data/filtered")
