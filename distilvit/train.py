import os
from functools import partial
import torch
from collections.abc import Mapping
import argparse

import nltk
import evaluate
import numpy as np
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    VisionEncoderDecoderModel,
    AutoFeatureExtractor,
    AutoTokenizer,
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import concatenate_datasets, DatasetDict


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


from .coco import get_dataset as get_coco_dataset
from .flickr import get_dataset as get_flickr_dataset


MAX_LENGTH = 128
CHECKPOINTS_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

SAVE_PATH = "../distilvit-flickr"


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
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
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


def data_collator(tokenizer, features):
    # XXX change this so it also works with flickr's labels_0, labels_1 etc
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}
    if "label" in first and first["label"] is not None:
        label = (
            first["label"].item()
            if isinstance(first["label"], torch.Tensor)
            else first["label"]
        )
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = (
                torch.long if isinstance(first["label_ids"][0], int) else torch.float
            )
            batch["labels"] = torch.tensor(
                [f["label_ids"] for f in features], dtype=dtype
            )

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                # make sure we pad or truncate
                if k == "labels":
                    truncated_features = []
                    for f in features:
                        item = f[k]
                        if len(item) != MAX_LENGTH:
                            print(
                                f"Found item of size {len(item)}), truncating or padding"
                            )
                            if len(item) > MAX_LENGTH:
                                item = item[:MAX_LENGTH]
                            else:
                                item = item + [tokenizer.pad_token_id] * (
                                    MAX_LENGTH - len(item)
                                )

                            assert len(item) == MAX_LENGTH

                        truncated_features.append(item)

                    batch[k] = torch.tensor(truncated_features)
                else:
                    batch[k] = torch.tensor([f[k] for f in features])
    return batch


_DATASETS = {"coco": get_coco_dataset, "flickr": get_flickr_dataset}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Vision Encoder Decoder Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--encoder-model",
        default="google/vit-base-patch16-224-in21k",
        type=str,
        help="Base model for the encoder",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        type=str,
        help="Base model to train again from",
    )
    parser.add_argument(
        "--feature-extractor-model",
        default="google/vit-base-patch16-224-in21k",
        type=str,
        help="Feature extractor model for the encoder",
    )
    parser.add_argument(
        "--decoder-model",
        default="distilbert/distilgpt2",
        type=str,
        help="Model for the decoder",
    )
    parser.add_argument(
        "--dataset",
        default="coco",
        choices=["coco", "flickr", "both"],
        help="Dataset to use for training",
    )
    return parser.parse_args()


def train(args):
    metric = evaluate.load("rouge")
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        args.feature_extractor_model
    )
    if args.base_model:
        model = VisionEncoderDecoderModel.from_pretrained(args.base_model)
    else:
        model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            args.encoder_model, args.decoder_model
        )
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.decoder_model)
    # GPT2 only has bos/eos tokens but not decoder_start/pad tokens
    tokenizer.pad_token = tokenizer.eos_token

    # update the model config
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    if args.dataset == "both":
        ds = _DATASETS["coco"](args.feature_extractor_model, args.decoder_model)
        ds2 = _DATASETS["flickr"](args.feature_extractor_model, args.decoder_model)
        combined = DatasetDict()
        for split in ds.keys():
            combined[split] = concatenate_datasets([ds[split], ds2[split]])
        ds = combined.shuffle(seed=42)
    else:
        ds = _DATASETS[args.dataset](args.feature_extractor_model, args.decoder_model)

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        output_dir=CHECKPOINTS_DIR,
        save_total_limit=10,
    )

    last_checkpoint = get_last_checkpoint(CHECKPOINTS_DIR)

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=feature_extractor,
        args=training_args,
        compute_metrics=partial(compute_metrics, tokenizer, metric),
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=partial(data_collator, tokenizer),
    )
    if last_checkpoint is not None:
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()
    trainer.save_model(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
