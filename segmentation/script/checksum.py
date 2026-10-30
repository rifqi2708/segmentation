from pathlib import Path
import hashlib
import json


def sha256_file(file_path):
    checksum = hashlib.sha256()

    # Read in small chunks to avoid loading the whole file into memory.
    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            checksum.update(chunk)

    return checksum.hexdigest()


root = Path("/workspace/segmentation/data/QUADRA_HC_MASK")
manifest_path = root / "sha256_manifest.json"

if not root.is_dir():
    raise FileNotFoundError(f"Folder does not exist: {root}")

mask_files = sorted(
    path for path in root.rglob("*.nii.gz") if path.is_file()
)

if not mask_files:
    raise RuntimeError(f"No .nii.gz masks found in {root}")

checksums = {}

for index, mask_path in enumerate(mask_files, start=1):
    relative_path = mask_path.relative_to(root).as_posix()
    checksums[relative_path] = sha256_file(mask_path)

    print(f"[{index}/{len(mask_files)}] Hashed: {relative_path}")

manifest = {
    "algorithm": "sha256",
    "file_count": len(checksums),
    "files": checksums,
}

with manifest_path.open("w", encoding="utf-8") as file:
    json.dump(manifest, file, indent=2)
    file.write("\n")

print(f"\nSaved checksums for {len(checksums)} masks.")
print(f"JSON: {manifest_path}")