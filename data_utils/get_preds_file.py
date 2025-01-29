import json
from sklearn.metrics import classification_report


# base, eq, noise, pitch, reverb, stretch, udio
# specnn_amplitude_base_mistral-tinyllama-train125, specnn_amplitude_base_wizardlm2-mistral-train125, specnn_amplitude_base_wizardlm2-tinyllama-train125
MODEL = "specnn_amplitude_base_wizardlm2-tinyllama-train125"
print(MODEL)

TYPE = "UDIO" if MODEL == "udio" else "SUNO"


# Load the dataset from a JSON file
if TYPE == "SUNO":
    dataset_file = "/data/nfs/analysis/interns/mfrohmann/robust-ai-lyrics-det/data/generated/suno/aggregated_sources_NLP4MUSA_SUNO-final.json"
elif TYPE == "UDIO":
    dataset_file = "/data/nfs/analysis/interns/mfrohmann/robust-ai-lyrics-det/data/generated/udio/aggregated_sources_NLP4MUSA.json"

with open(dataset_file, "r") as f:
    main_data = json.load(f)

# Load the predictions
with open(
    f"/data/nfs/analysis/interns/mfrohmann/deepfake-detector/predictions/{MODEL}.json",
    "r",
) as f:
    predictions_data = json.load(f)

# Create a lookup dictionary from predictions data
predictions_lookup = {
    path.replace(".mp3", "").replace("0.", ""): [prediction, binary_prediction]
    for path, prediction, binary_prediction in zip(
        predictions_data["paths"], predictions_data["predictions"], predictions_data["binary_predictions"]
    )
}

# Extend the main database with predictions
for idx, item in enumerate(main_data["test"]):
    if "md5" in item.keys():
        md5 = item["md5"]
    else:
        if item["mp3_urls"] is None or item["mp3_urls"][0] is None:
            continue
        if TYPE == "SUNO":
            md5 = item["mp3_urls"][0].split("/")[-1].replace(".mp3", "")
        elif TYPE == "UDIO":
            md5 = item["label_model"] + "_" + item["merged_md5"]
    if md5 in predictions_lookup:
        item["binary_prediction"] = predictions_lookup[md5][1]
        item["prediction"] = predictions_lookup[md5][0]
    else:
        # print(idx)
        item["binary_prediction"] = None  # Or a default value like -1


# Create the new JSON structure
output_data = {
    "predictions": [],
    "probabilities": [],
    "references": [],
    "model_names": [],
    "artists": [],
    "langs": [],
    "genres": [],
    "lyrics_ids": [],
}

# Populate the lists
for idx, item in enumerate(main_data["test"]):
    # predictions
    # if no prediction here, skip
    if "prediction" not in item.keys() or item["prediction"] is None:
        # print(idx)
        continue
    # flip binary precictions
    output_data["predictions"].append(1 if item["binary_prediction"] == 0 else 0)
    # flip predictions
    output_data["probabilities"].append(1 - item["prediction"])
    # probabilities

    # references
    output_data["references"].append(0 if item["label_model"] == "original" else 1)

    # model_names
    output_data["model_names"].append(item["label_model"])

    # artists
    output_data["artists"].append(
        item["artist_name"] if item["label_model"] == "original" else "generated"
    )

    # langs
    output_data["langs"].append(item["label_lang"])

    # genres
    output_data["genres"].append(item["label_genre"])

    # lyrics_ids
    if item["label_model"] == "original":
        output_data["lyrics_ids"].append(item["md5"])
    else:
        output_data["lyrics_ids"].append("merged_md5")


cr = classification_report(
    output_data["references"], output_data["predictions"], output_dict=False, digits=4
)

print(cr)
# Save the new JSON
with open(
    f"/data/nfs/analysis/interns/mfrohmann/deepfake-detector/predictions/transformed/{MODEL}.json",
    "w",
) as f:
    json.dump(output_data, f, indent=4)

print(
    f"Transformed data saved to /data/nfs/analysis/interns/mfrohmann/deepfake-detector/predictions/transformed/{MODEL}.json"
)
