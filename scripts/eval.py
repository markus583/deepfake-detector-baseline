"""
Simple evaluation code

example:
python -m scripts.final.eval --config specnn_amplitude --gpu 0 --steps 20 --repeat 5
"""

import argparse
import json
import os
import sys

import tensorflow as tf
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from loader.audio import AdversarialAugmenter, AudioLoader, EvalAugmenter
from loader.config import ConfLoader
from loader.global_variables import *
from model.simple_cnn import SimpleCNN, SimpleSpectrogramCNN

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", help="config file", type=str, default="specnn_amplitude"
)
parser.add_argument("--weights", help="weights file, else defaults to config", type=str)
parser.add_argument(
    "--encoder", help="eval model trained on only one encoder", type=str, default=""
)
parser.add_argument("--gpu", help="gpu to evaluate on", type=int, default=0)
parser.add_argument("--steps", help="gpu to evaluate on", type=int, default=500)
parser.add_argument("--repeat", help="gpu to evaluate on", type=int, default=-1)
parser.add_argument("--codec", help="codec", type=str, default="")
parser.add_argument("--external_home", action="store_true")
parser.add_argument("--data_setup", type=str, default="stratify_3")
parser.add_argument("--adversarial", type=str, default="")
args = parser.parse_args()

if not args.weights:
    args.weights = args.config
GPU = args.gpu
CODEC = ""
CODEC_EXTENSION = ""

if args.codec != "":
    CODEC_EXTENSION = args.codec
    CODEC = CODEC_EXTENSION + "_64"
    print("\nEvaluating on codec {}!\n".format(CODEC))
    if args.codec == "opus":
        CODEC = "libopus"
    POS_DB_PATH = CODEC_DB_PATH + CODEC + "/real"
    NEG_DB_PATH = CODEC_DB_PATH + CODEC + "/"

ENCODER = ""
ADVERSARIAL = args.adversarial
print(ENCODER, ADVERSARIAL)

if args.encoder:
    ENCODER = args.encoder

configuration = ConfLoader(CONF_PATH)
configuration.load_model(args.config)
global_conf = configuration.conf

gpus = tf.config.list_physical_devices("GPU")
tf.config.set_visible_devices(gpus[GPU], "GPU")
tf.config.experimental.set_memory_growth(gpus[GPU], True)

if args.data_setup == "stratify_1":
    NEG_DB_PATH = HOME + "data/fake/stratify/mistral-tinyllama-train125/128/0"
    WEIGHTS_PATH = HOME + "/weights/final/stratify/mistral-tinyllama-train125/128"
elif args.data_setup == "stratify_2":
    NEG_DB_PATH = HOME + "data/fake/stratify/wizardlm2-mistral-train125/128/0"
    WEIGHTS_PATH = HOME + "/weights/final/stratify/wizardlm2-mistral-train125/128"
elif args.data_setup == "stratify_3":
    NEG_DB_PATH = HOME + "data/fake/stratify/wizardlm2-tinyllama-train125/128/0"
    WEIGHTS_PATH = HOME + "/weights/final/stratify/wizardlm2-tinyllama-train125/128"
elif args.data_setup != "":
    raise ValueError("Unknown data setup")

if args.repeat > 0:
    global_conf["repeat"] = args.repeat  # keep the repeat low

global_conf["batch_size"] = 1

##


if not ADVERSARIAL:
    OUT_NAME = "base"
    adversarial = None
elif ADVERSARIAL == "udio":
    OUT_NAME = "udio"
    adversarial = None
    POS_DB_PATH = HOME + "/data/udio/real"
    NEG_DB_PATH = HOME + "/data/udio/fake/128/0"

else:
    OUT_NAME = ADVERSARIAL

    adversarial = AdversarialAugmenter([ADVERSARIAL])
if args.data_setup:
    if args.data_setup == "stratify_1":
        OUT_NAME += "_mistral-tinyllama-train125"
    elif args.data_setup == "stratify_2":
        OUT_NAME += "_wizardlm2-mistral-train125"
    elif args.data_setup == "stratify_3":
        OUT_NAME += "_wizardlm2-tinyllama-train125"
print(OUT_NAME)
##

loader = AudioLoader(
    POS_DB_PATH,
    NEG_DB_PATH,
    global_conf,
    split_path=SPLIT_PATH,
    codec=CODEC_EXTENSION,
)
augmenter = EvalAugmenter(global_conf)


@tf.function
def one_hot_encoder(y, depth):
    y1, y2 = y
    y1 = tf.cast(y1, tf.int32)
    idx = (1 - y1) * (y2 + 1)
    return y1, tf.one_hot(idx, depth)


it_test = loader.create_tf_iterator(
    "test", augmenter=augmenter, adversarial=adversarial
)
if not ENCODER:
    it_test = it_test.map(lambda x, y: (x, one_hot_encoder(y, loader.n_encoders + 1)))
else:
    it_test = it_test.map(lambda x, y: (x, y[0]))

if "use_raw" in global_conf:
    it_test = it_test.map(lambda x, y: (tf.expand_dims(x, -1), y))

_it = iter(it_test)
input_batch, y_batch = next(_it)
print(input_batch.shape[1:])
input_size = input_batch.shape[1:]

## normal run

n_encoders = loader.n_encoders + 1
if ENCODER:
    print("\nEncoder-specific model! `{}`\n".format(ENCODER))
    n_encoders = None
if "use_raw" in global_conf:
    model = SimpleCNN(input_size, global_conf, detect_encoder=n_encoders)
else:
    model = SimpleSpectrogramCNN(input_size, global_conf, detect_encoder=n_encoders)

# model.m.summary()
model.m.load_weights(os.path.join(WEIGHTS_PATH, args.weights + args.encoder))

# --- Evaluation and Prediction Generation ---
scores = {}
binary_predictions_list = []
predictions_list = []
y_list = []
test_paths = []
test_iter = iter(it_test)
# TODO: batch this? faster or not?

print("\nEvaluating on the test set:")
for i, step in tqdm(enumerate(range(7000)), total=7000):
    try:
        x_batch, y_batch = next(test_iter)
        path = loader.split_mp3["test"][i]

        # Get model predictions
        predictions = model.m.predict(x_batch, verbose=0)

        # Convert probabilities to binary predictions (0 or 1)
        binary_predictions = (predictions[0] > 0.5).astype(int)

        # Flatten the predictions to a 1D array and add to the list
        binary_predictions_list.extend(binary_predictions.flatten().tolist())
        predictions_list.extend(predictions[0].flatten().tolist())
        test_paths.append(path)
        y_list.extend(y_batch[0].numpy().tolist())
        if binary_predictions_list[-1] != y_list[-1]:
            print(i, y_list[-1], "FAIL")
    except StopIteration:
        print("DONE: ", i)
        break

print(f"Generated predictions for {len(binary_predictions_list)} samples.")

# print total accuracy
total = len(binary_predictions_list)
correct = sum([1 for i in range(total) if binary_predictions_list[i] == y_list[i]])
print(f"Total accuracy: {correct / total}")
print("DONE")

# save to json
with open(f"predictions/{args.config}_{OUT_NAME}.json", "w") as f:
    json.dump(
        {
            "paths": test_paths,
            "predictions": predictions_list,
            "binary_predictions": binary_predictions_list,
        },
        f,
        indent=4,
    )
