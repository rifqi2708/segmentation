import SimpleITK as sitk
import numpy as np
from pathlib import Path
import json


output_dir = Path("/workspace/segmentation/data/selected_organ")

#writing json report
def write_report_json (parameters):
    json_dir = "/workspace/segmentation/data/output/component_count.json"
    json_path = Path(json_dir)

    #get the written parameters
    if json_path.exists() and json_path.stat().st_size > 0 :
        with open (json_path, "r", encoding="utf-8") as file:
            try:
                current_data = json.load(file)
                if not isinstance(current_data, list):
                    current_data = []
            except json.JSONDecodeError:
                current_data = []
    else:
        current_data = []

    # add the current parameters to the written parameters (to update the parameters)
    current_data.append(parameters)

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(current_data, file, indent=4)


region_size = {}

for subject_dir in sorted(output_dir.iterdir()): #iterate through all subjects
    if not subject_dir.is_dir():
        continue 

    subject_name = subject_dir.name
    region_per_subject = {}
    
    for session_dir in sorted(subject_dir.iterdir()): #iterate thorugh all sessions
        if not session_dir.is_dir():
            continue

        session_name = session_dir.name
        region_per_session = {}
        mask_dirs = sorted(session_dir.glob("*.nii.gz"))

        for mask_dir in mask_dirs:
            mask_name = mask_dir.name.removesuffix(".nii.gz")
            mask = sitk.ReadImage(mask_dir)
            components = sitk.ConnectedComponent(mask, False)
            inspector = sitk.RelabelComponentImageFilter()
            inspector.Execute(components)
            component_size = inspector.GetSizeOfObjectsInPixels()

            blob = max(component_size) * 0.01 
            threshold = max(blob, 3) 

            filter = sitk.RelabelComponentImageFilter()
            filter.SetMinimumObjectSize(int(threshold))
            retained_components = filter.Execute(components)
            filtered_mask = retained_components > 0

            filtered_component_size = filter.GetSizeOfObjectsInPixels()

            filtered_dir = Path(*("filtered" if part == "selected_organ" else part for part in mask_dir.parts))
            filtered_dir.parent.mkdir(parents=True, exist_ok=True)

            sitk.WriteImage(filtered_mask, filtered_dir)

            region_per_session[mask_name] = filtered_component_size

        region_per_subject[session_name] = region_per_session
        print (f"{subject_name} - {session_name} >>> DONE")

    region_size[subject_name] = region_per_subject



write_report_json(region_size)