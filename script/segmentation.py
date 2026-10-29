from totalsegmentator.python_api import totalsegmentator
from pathlib import Path
import yaml
import time
import json

# get list of organ and task required in this project form organs.yaml
def get_organ_task (organ_yaml: str):
    with open(organ_yaml,"r") as f:
        organs = yaml.safe_load(f)
        
    organs_in_total = []
    organs_in_head = []
    all_organs = []
    tasks = set()
    
    for organ in organs["organs"]:
        all_organs.append(organ["source_class"]) #get all required organ 
        tasks.add(organ["task"])
        
        if (organ["task"]) == "total":
            organs_in_total.append(organ["source_class"]) #get organ in total (for roi_subset)
        else:
            organs_in_head.append(organ["source_class"]) #get organ in head_glands_cavities (for masks selection)
            
    return all_organs, organs_in_total, organs_in_head, tasks   
    #list of all organ, organ in total (for roi_subset), organ in head_glands_cavities (for masks selection)

# get parameters for segmentation
def get_params (data_path: str, organ_yaml: str):
    data_folder = Path(data_path)
    input_folder = data_folder / "input"
    output_folder = data_folder / "spinal_cord" 
    params = []
    _, roi_subset, _, tasks = get_organ_task(organ_yaml) # get tasks and roi_subset
    
    for subject_dir in sorted(input_folder.iterdir()): #get subjects in input folder Subject = quadra 1-48
        if subject_dir.is_dir():
            subject_img_dirs = sorted(subject_dir.glob("*.nii.gz"))
            
            if not subject_img_dirs:
                raise ValueError (f"WARNING!!! images not found in this dir: '{subject_dir}'") 
            else:

                #get the output dir for TotalSegmentator
                for subject_img_dir in subject_img_dirs: 
                    session_name = subject_img_dir.name.removesuffix("_CT-AC.nii.gz") # get session_name --> Session = test, retest
                    subject_mask_dir = output_folder / subject_dir.name / session_name # formulating output dirs for TotalSegmentator
                    subject_mask_dir.mkdir(parents=True, exist_ok=True) #creating folder based on formulated dirs
                    
                    #listing the parameter
                    for task in tasks:
                        param = {
                            "subject_id": subject_dir.name,
                            "session": session_name,
                            "task": task,
                            "input": subject_img_dir,
                            "output": subject_mask_dir,
                            "device": "gpu"
                        }
                        if task == "total":
                            param["roi_subset"] = roi_subset #add roi_subset if the task is "toal"
                        params.append(param)
    
    
    return params #list of dict{parameter} (subject, session, task, input dir, output dir, device)


#writing json report
def write_report_json (parameters):
    json_dir = "/workspace/segmentation/data/spinal_cord/segmentation_parameters.json"
    json_path = Path(json_dir)
    
    #write the current parameters
    parameter_json = {
        "subject": parameters["subject_id"],
        "session": parameters["session"],
        "input": str(parameters["input"]),
        "output": str(parameters["output"]),
        "task": parameters["task"],
        "roi_subset": parameters.get("roi_subset", None),
        "device": parameters["device"]
    }

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
    current_data.append(parameter_json)

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(current_data, file, indent=4)


# main function to run the segmentation
def run_segmentation (data_path: str, organ_yaml: str):
    parameters = get_params (data_path, organ_yaml)
    start_time = time.perf_counter()
    
    for parameter in parameters:

        #run the segmentation
        
        totalsegmentator(
            input = parameter["input"],
            output = parameter["output"],
            task = parameter["task"],
            roi_subset = parameter.get("roi_subset", None),
            device = parameter["device"],
            nr_thr_resamp=1,
            nr_thr_saving=1,
        )

        write_report_json(parameter) # write the json file after one segmentation run

    end_time = time.perf_counter()
    execution_time = (end_time - start_time)/60 # calculating the running time
    print(f"Segmentation time: {execution_time:.6f} minutes")
    
    
# Running the segmentation
if __name__ == "__main__":
    run_segmentation(
        "/workspace/segmentation/data",
        "/workspace/segmentation/spinal_cord.yaml"
        )