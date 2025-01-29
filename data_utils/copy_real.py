import json
import os
import shutil


# Generate song path from md5 hash and quality
def py_song_path_from_md5(md5: str, quality: str = "mp3_128") -> str:
    """Generate the song path based on the md5 hash and quality."""
    return "/data/music/output/{}/{}/{}/{}/{}.mp3".format(
        quality, md5[0], md5[1], md5[2], md5
    )


# Load the dataset from a JSON file (replace 'your_dataset.json' with your actual file)
TYPE = "UDIO"

# Load the dataset from a JSON file
if TYPE == "SUNO":
    dataset_file = "/data/nfs/analysis/interns/mfrohmann/robust-ai-lyrics-det/data/generated/suno/aggregated_sources_NLP4MUSA_SUNO-final.json"
    base_dest_dir = "/data/nfs/analysis/interns/mfrohmann/deepfake-detector/data/real"

elif TYPE == "UDIO":
    dataset_file = "/data/nfs/analysis/interns/mfrohmann/robust-ai-lyrics-det/data/generated/udio/aggregated_sources_NLP4MUSA.json"
    base_dest_dir = (
        "/data/nfs/analysis/interns/mfrohmann/deepfake-detector/data/udio/real"
    )

with open(dataset_file, "r") as f:
    dataset = json.load(f)


# Create destination directories if they don't exist
os.makedirs(os.path.join(base_dest_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(base_dest_dir, "test"), exist_ok=True)

# Iterate through the training dataset
for split in ["train", "test"]:
    for song in dataset[split]:
        if song["label_model"] == "original":
            md5 = song["md5"]
            source_path = py_song_path_from_md5(md5)

            # Determine destination subdirectory based on label_subset
            if song["label_subset"] == "train":
                dest_subdir = "train"
            elif song["label_subset"] == "test":
                dest_subdir = "test"
            else:
                continue  # Skip if label_subset is not train or test

            dest_path = os.path.join(base_dest_dir, dest_subdir, f"{md5}.mp3")

            # Copy the song if the source file exists
            if os.path.exists(source_path):
                shutil.copyfile(source_path, dest_path)
                print(f"Copied {source_path} to {dest_path}")
            else:
                print(f"Source file not found: {source_path}")

print("Finished copying songs.")
