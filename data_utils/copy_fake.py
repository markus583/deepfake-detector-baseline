import json
import os
import shutil
import re

# SUNO, UDIO, STRATIFY_X
TYPE = "STRATIFY_MISTRAL_TINYLLAMA"
TYPE = "STRATIFY_WIZARDLM_MISTRAL"
TYPE = "STRATIFY_WIZARDLM_TINYLLAMA"


# Load the dataset from a JSON file
if TYPE == "SUNO":
    dataset_file = "/data/nfs/analysis/interns/mfrohmann/robust-ai-lyrics-det/data/generated/suno/aggregated_sources_NLP4MUSA_SUNO-final.json"
    base_dest_dir = (
        "/data/nfs/analysis/interns/mfrohmann/deepfake-detector/data/fake"
    )

elif TYPE == "UDIO":
    dataset_file = "/data/nfs/analysis/interns/mfrohmann/robust-ai-lyrics-det/data/generated/udio/aggregated_sources_NLP4MUSA.json"
    base_dest_dir = "/data/nfs/analysis/interns/mfrohmann/deepfake-detector/data/udio/fake"
elif TYPE == "STRATIFY_MISTRAL_TINYLLAMA":
    dataset_file = "/data/nfs/analysis/interns/mfrohmann/robust-ai-lyrics-det/data/generated/suno/aggregated_sources_NLP4MUSA_SUNO-final_mistral-tinyllama-train125.json"
    base_dest_dir = "/data/nfs/analysis/interns/mfrohmann/deepfake-detector/data/fake/stratify/mistral-tinyllama-train125"
elif TYPE == "STRATIFY_WIZARDLM_MISTRAL":
    dataset_file = "/data/nfs/analysis/interns/mfrohmann/robust-ai-lyrics-det/data/generated/suno/aggregated_sources_NLP4MUSA_SUNO-final_wizardlm2-mistral-train125.json"
    base_dest_dir = "/data/nfs/analysis/interns/mfrohmann/deepfake-detector/data/fake/stratify/wizardlm2-mistral-train125"
elif TYPE == "STRATIFY_WIZARDLM_TINYLLAMA":
    dataset_file = "/data/nfs/analysis/interns/mfrohmann/robust-ai-lyrics-det/data/generated/suno/aggregated_sources_NLP4MUSA_SUNO-final_wizardlm2-tinyllama-train125.json"
    base_dest_dir = "/data/nfs/analysis/interns/mfrohmann/deepfake-detector/data/fake/stratify/wizardlm2-tinyllama-train125"
    

with open(dataset_file, "r") as f:
    dataset = json.load(f)

# Define base destination directory

# Create destination directories if they don't exist
for i in range(2):  # Assuming a maximum of 2 indices in mp3_urls
    os.makedirs(os.path.join(base_dest_dir, "128", str(i), "train"), exist_ok=True)
    os.makedirs(os.path.join(base_dest_dir, "128", str(i), "test"), exist_ok=True)

# Iterate through the dataset
n_songs_copied = 0
id_list = []
for split in ["train", "test"]:
    for song_idx, song in enumerate(dataset[split]):
        if song["label_model"] != "original":
            # Extract UUIDs from mp3_urls using regular expressions
            if song["mp3_urls"] is None or song["mp3_urls"] == [None, None]:
                print(f"Song: {song['merged_md5']} skipped")
                continue
            for idx, url in enumerate(song["mp3_urls"]):
                if TYPE != "UDIO":
                    match = re.search(
                        r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
                        url,
                    )

                    if match:
                        uuid = match.group(1)
                        source_path = os.path.join(
                            "/data/nfs/analysis/interns/mfrohmann/robust-ai-lyrics-det/output/128kbps/mp3",
                            f"{uuid}.mp3",
                        )

                        # Determine destination subdirectory based on label_subset
                        if split == "train":
                            dest_subdir = "train"
                        elif split == "test":
                            dest_subdir = "test"
                        else:
                            continue  # Skip if label_subset is not train or test

                        dest_path = os.path.join(
                            base_dest_dir, "128", str(idx), dest_subdir, f"{uuid}.mp3"
                        )

                        # Copy the song if the source file exists
                        if os.path.exists(source_path):
                            shutil.copyfile(source_path, dest_path)
                            if split == "train":
                                n_songs_copied += 1
                                id_list.append(uuid)
                            print(f"Copied {source_path} to {dest_path}")
                        else:
                            if idx != 1:
                                print(f"Source file not found: {source_path}")
                    else:
                        print(f"Could not extract UUID from URL: {url}")
                else:
                    source_path = os.path.join(
                        "/data/nfs/analysis/interns/mfrohmann/robust-ai-lyrics-det",
                        url,
                    )
                    print(source_path)
                    # Determine destination subdirectory based on label_subset
                    if song["label_subset"] == "train":
                        dest_subdir = "train"
                    elif song["label_subset"] == "test":
                        dest_subdir = "test"
                    else:
                        continue  # Skip if label_subset is not train or test

                    dest_path = os.path.join(
                        base_dest_dir,
                        "128",
                        str(idx),
                        dest_subdir,
                        url.split("/")[-2] + ".mp3",
                    )

                    # Copy the song if the source file exists
                    if os.path.exists(source_path):
                        shutil.copyfile(source_path, dest_path)
                        print(f"Copied {source_path} to {dest_path}")
                    else:
                        print(f"Source file not found: {source_path}")

print("Finished copying songs.")
print(n_songs_copied, len(id_list), len(set(id_list)))
