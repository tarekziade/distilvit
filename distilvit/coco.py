"""
Downloads COCO and creates a tokenized one with extracted features.
"""
import requests
import os
from functools import partial
from collections.abc import Mapping

import nltk
import numpy as np
from PIL import Image
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from transformers import (
    AutoFeatureExtractor,
    AutoTokenizer,
)


try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    nltk.download("punkt", quiet=True)


def download_file(url, directory):
    local_filename = url.split("/")[-1]
    os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
    path_to_file = os.path.join(directory, local_filename)

    # Only download if the file does not exist
    if not os.path.exists(path_to_file):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size_in_bytes = int(r.headers.get("content-length", 0))
            block_size = 1024  # 1 Kibibyte
            progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
            with open(path_to_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=block_size):
                    progress_bar.update(len(chunk))
                    f.write(chunk)
            progress_bar.close()
        return path_to_file
    else:
        print(f"{local_filename} already exists. Skipping download.")
        return path_to_file


urls = [
    "http://images.cocodataset.org/zips/train2017.zip",
    "http://images.cocodataset.org/zips/val2017.zip",
    "http://images.cocodataset.org/zips/test2017.zip",
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "http://images.cocodataset.org/annotations/image_info_test2017.zip",
]

COCO_DIR = os.path.join(os.path.dirname(__file__), "..", "coco")
CACHED_DS = "/media/user/Extreme SSD/cache/coco"
# os.path.join(os.path.dirname(__file__), "..", "cache", "coco")
MAX_LENGTH = 128
CHECKPOINTS_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
SAVE_PATH = "./distilvit"


def tokenization_fn(tokenizer, captions):
    """Run tokenization on captions."""
    labels = tokenizer(captions, padding="max_length", max_length=MAX_LENGTH).input_ids

    return labels


def extract_features(feature_extractor, image_paths):
    images = []
    for image_path in image_paths:
        try:
            images.append(Image.open(image_path).convert("RGB"))
        except Exception:
            pass

    inputs = feature_extractor(images=images, return_tensors="pt")
    for image in images:
        image.close()
    return inputs


def preprocess_fn(
    feature_extractor,
    tokenizer,
    examples,
):
    """Run tokenization + image feature extraction"""
    image_paths = examples["image_path"]
    captions = examples["caption"]
    model_inputs = {}
    model_inputs["labels"] = tokenization_fn(tokenizer, captions)
    model_inputs["pixel_values"] = extract_features(
        feature_extractor,
        image_paths,
    )["pixel_values"]

    return model_inputs


def get_dataset(feature_extractor_model, text_decoder_model):
    """Downloads the COCO dataset and tokenizes it.

    The result is saved on disk so we can reuse it.
    """
    if os.path.exists(CACHED_DS):
        return load_from_disk(CACHED_DS)

    feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_model)
    tokenizer = AutoTokenizer.from_pretrained(text_decoder_model)
    tokenizer.pad_token = tokenizer.eos_token

    for url in urls:
        print(f"Downloading {url}...")
        download_file(url, COCO_DIR)
    print("Download complete.")

    ds = load_dataset(
        "ydshieh/coco_dataset_script",
        "2017",
        data_dir=COCO_DIR,
        trust_remote_code=True,
    )
    ds = ds.map(
        function=partial(preprocess_fn, feature_extractor, tokenizer),
        batched=True,
        remove_columns=ds["train"].column_names,
    )
    # save the mapped dataset so we can reuse it
    ds.save_to_disk(CACHED_DS)

    return ds
