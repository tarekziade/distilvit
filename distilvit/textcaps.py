"""
Tokenizes the TextCaps dataset
"""
from datasets import load_dataset, load_from_disk, concatenate_datasets
from functools import partial
import os
from transformers import (
    AutoTokenizer,
    AutoFeatureExtractor,
)


MAX_LENGTH = 128


def tokenization_fn(tokenizer, captions):
    return tokenizer(captions, padding="max_length", max_length=MAX_LENGTH).input_ids


def preprocess_fn(
    idx,
    feature_extractor,
    tokenizer,
    examples,
):
    model_inputs = {}
    captions = [cap[idx] for cap in examples["caption_str"]]
    model_inputs["labels"] = tokenization_fn(tokenizer, captions)

    model_inputs["pixel_values"] = feature_extractor(
        images=examples["image"], return_tensors="pt"
    )["pixel_values"]
    return model_inputs


def get_dataset(feature_extractor_model, text_decoder_model):
    cached_ds = os.path.join(cache_dir, "textcaps")
    if os.path.exists(cached_ds):
        return load_from_disk(cached_ds)

    ds = load_dataset("lmms-lab/TextCaps", split="train")

    feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_model)
    tokenizer = AutoTokenizer.from_pretrained(text_decoder_model)
    tokenizer.pad_token = tokenizer.eos_token

    batches = []
    for idx in range(5):
        batches.append(
            ds.map(
                function=partial(preprocess_fn, idx, feature_extractor, tokenizer),
                batched=True,
                remove_columns=ds.column_names,
            )
        )

    ds = concatenate_datasets(batches)
    # splitting in train (80%), test (10%), validation (10%)
    ds = ds.train_test_split(test_size=0.2)
    test_and_eval = ds["test"].train_test_split(test_size=0.5)
    ds["test"] = test_and_eval["test"]
    ds["validation"] = test_and_eval["train"]
    ds.save_to_disk(CACHED_DS)
    return ds
