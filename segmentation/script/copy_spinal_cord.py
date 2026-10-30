from pathlib import Path
import shutil

source_dir = Path("/workspace/segmentation/data/spinal_cord")
destination_dir = Path("/workspace/segmentation/data/selected_organ")

if not source_dir.is_dir():
    raise FileNotFoundError(f"Source folder does not exist: {source_dir}")

copied = 0
skipped = 0

for source_file in sorted(source_dir.glob("*/*/spinal_cord.nii.gz")):
    if not source_file.is_file():
        continue

    # Example: QUADRA_HC_001/retest/spinal_cord.nii.gz
    relative_path = source_file.relative_to(source_dir)
    destination_file = destination_dir / relative_path

    # Create the subject/session folders if needed.
    destination_file.parent.mkdir(parents=True, exist_ok=True)

    # Keep existing files unchanged.
    if destination_file.exists():
        print(f"Skipped (already exists): {relative_path}")
        skipped += 1
        continue

    shutil.copy2(source_file, destination_file)
    print(f"Copied: {relative_path}")
    copied += 1

print(f"\nFinished: {copied} copied, {skipped} skipped.")