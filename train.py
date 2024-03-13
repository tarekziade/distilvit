import requests
import os
from functools import partial
import torch

import nltk
import evaluate
import numpy as np
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    default_data_collator,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    VisionEncoderDecoderModel,
    AutoFeatureExtractor,
    AutoTokenizer,
)


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA (Nvidia GPU).")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU).")
else:
    device = torch.device("cpu")
    print("Using CPU.")


os.environ["WANDB_DISABLED"] = "true"

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

COCO_DIR = "coco"
ignore_pad_token_for_loss = True


def tokenization_fn(tokenizer, captions, max_target_length):
    """Run tokenization on captions."""
    labels = tokenizer(
        captions, padding="max_length", max_length=max_target_length
    ).input_ids

    return labels


# image preprocessing step
def feature_extraction_fn(feature_extractor, image_paths, check_image=True):
    """
    Run feature extraction on images
    If `check_image` is `True`, the examples that fails during `Image.open()` will be caught and discarded.
    Otherwise, an exception will be thrown.
    """
    if check_image:
        images = []
        to_keep = []
        for image_file in image_paths:
            try:
                img = Image.open(image_file)
                images.append(img)
                to_keep.append(True)
            except Exception:
                to_keep.append(False)
    else:
        images = [Image.open(image_file) for image_file in image_paths]

    encoder_inputs = feature_extractor(images=images, return_tensors="np")

    return encoder_inputs.pixel_values


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
        feature_extractor, image_paths, check_image=check_image
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

    for url in urls:
        print(f"Downloading {url}...")
        download_file(url, COCO_DIR)
    print("Download complete.")

    ds = load_dataset("ydshieh/coco_dataset_script", "2017", data_dir=COCO_DIR)
    ds = ds.map(
        function=partial(preprocess_fn, feature_extractor),
        batched=True,
        fn_kwargs={"max_target_length": 128},
        remove_columns=ds["train"].column_names,
    )

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        output_dir="./image-captioning-output",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=feature_extractor,
        args=training_args,
        compute_metrics=partial(compute_metrics, tokenizer, metric),
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=default_data_collator,
    )

    trainer.train()
    trainer.save_model("./distilvit")
    tokenizer.save_pretrained("./distilvit")


if __name__ == "__main__":
    train()
