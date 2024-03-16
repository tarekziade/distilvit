import requests
import os
from functools import partial
import torch
import hashlib
from collections.abc import Mapping

import nltk
import evaluate
import numpy as np
from PIL import Image
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase, DataCollatorWithPadding
from transformers import (
    default_data_collator,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    VisionEncoderDecoderModel,
    AutoFeatureExtractor,
    AutoTokenizer,
)
from transformers.trainer_utils import get_last_checkpoint


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA (Nvidia GPU).")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU).")
else:
    device = torch.device("cpu")
    print("Using CPU.")


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

COCO_DIR = os.path.join(os.path.dirname(__file__), "coco")
CACHED_DS = os.path.join(os.path.dirname(__file__), "cache", "dataset")


ignore_pad_token_for_loss = True


def tokenization_fn(tokenizer, captions, max_target_length):
    """Run tokenization on captions."""
    labels = tokenizer(
        captions, padding="max_length", max_length=max_target_length
    ).input_ids

    return labels


def extract_features(image_paths, feature_extractor):
    images = []
    for image_path in image_paths:
        try:
            images.append(Image.open(image_path).convert("RGB"))
        except Exception:
            pass

    inputs = feature_extractor(images=images, return_tensors="pt")
    #torch.save(inputs, save_path)
    for image in images:
        image.close()
    return inputs


# XXX opens too many FDs, to fix
# image preprocessing step
def feature_extraction_fn(feature_extractor, image_paths):
    features = extract_features(image_paths, feature_extractor)
    return features['pixel_values']


def preprocess_fn(
    feature_extractor, tokenizer, examples, max_target_length, check_image=True
):
    """Run tokenization + image feature extraction"""
    image_paths = examples["image_path"]
    captions = examples["caption"]

    model_inputs = {}
    # This contains image path column
    model_inputs["labels"] = tokenization_fn(tokenizer, captions, max_target_length)
    model_inputs["pixel_values"] = feature_extraction_fn(
        feature_extractor, image_paths, 
    )
    return model_inputs


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels


def compute_metrics(tokenizer, metric, eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    if ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    return result


def torch_data_collator(tokenizer, features):
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if isinstance(first["label_ids"][0], int) else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                if k == "labels":
                    truncated_features = []
                    for f in features:
                        item = f[k]
                        if len(item) != 128:
                            print(f"Found item of size {len(item)}), truncating or padding")
                            if len(item) > 128:
                                item = item[:128]
                            else:
                                item = item + [tokenizer.pad_token_id] * (128-len(item))
                            
                            assert len(item) == 128

                        truncated_features.append(item)

                    batch[k] = torch.tensor(truncated_features)
                else:
                    batch[k] = torch.tensor([f[k] for f in features])
    return batch



def train():
    metric = evaluate.load("rouge")
    image_encoder_model = "google/vit-base-patch16-224-in21k"
    text_decode_model = "distilbert/distilgpt2"

    feature_extractor = AutoFeatureExtractor.from_pretrained(image_encoder_model)
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        image_encoder_model, text_decode_model
    )
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(text_decode_model)
    # GPT2 only has bos/eos tokens but not decoder_start/pad tokens
    tokenizer.pad_token = tokenizer.eos_token

    # update the model config
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id


    if os.path.exists(CACHED_DS):
        ds = load_from_disk(CACHED_DS)
    else:
        for url in urls:
            print(f"Downloading {url}...")
            download_file(url, COCO_DIR)
        print("Download complete.")

        ds = load_dataset("ydshieh/coco_dataset_script", "2017", data_dir=COCO_DIR, trust_remote_code=True)
        ds = ds.map(
            function=partial(preprocess_fn, feature_extractor, tokenizer),
            batched=True,
            fn_kwargs={"max_target_length": 128},
            remove_columns=ds["train"].column_names,
        )

        # save the mapped dataset so we can reuse it
        ds.save_to_disk(CACHED_DS)

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        output_dir="./image-captioning-output",
        save_total_limit=10
    )
    
    last_checkpoint = get_last_checkpoint("./image-captioning-output")

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=feature_extractor,
        args=training_args,
        compute_metrics=partial(compute_metrics, tokenizer, metric),
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=partial(torch_data_collator, tokenizer)
    )
    if last_checkpoint is not None:
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()
    trainer.save_model("./distilvit")
    tokenizer.save_pretrained("./distilvit")


if __name__ == "__main__":
    train()
