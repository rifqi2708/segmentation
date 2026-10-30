##Segmentaiton workflow

1. Download data using setup.sh

2. Make sure, data, input, output folders are exist

3. Run segmentation using : /workspace/segmentation/segmentation/script/segmentation.py
    adjust the data and organ.yaml dir

4. Select the required organ using: /workspace/segmentation/segmentation/notebook/selecting_required_masks.ipynb

5. Derive carnial and thoracic spinal cord using : /workspace/segmentation/segmentation/notebook/spinal_cord_selection.ipynb

6. Remove isolated voxel (1% from the bigest volume) using: /workspace/segmentation/segmentation/script/filtration.py

7. Do technical QA for the final masks using : /workspace/segmentation/segmentation/script/mask_qa.py
    - cek if :
        - masks readable
        - masks not empty
        - masks binary
        - masks 3D
        - masks have the same size, spacing, origin, direction with the images

done

* note: segmentation/script/copy_spinal_cord.py and config/spinal_cord.yaml used to segmenent the spinal cord because I was forgot to inclune it in the first segmentation
* note: checksum used to check download completeness, a separate script available locally


