"""
Tokenizes the Flickr30k dataset
"""
from datasets import load_dataset, load_from_disk, concatenate_datasets
from functools import partial
import os
from transformers import (
    AutoTokenizer,
    AutoFeatureExtractor,
)


MAX_LENGTH = 128
CACHED_DS = os.path.join(os.path.dirname(__file__), "..", "cache", "flickr30k")


def tokenization_fn(tokenizer, captions):
    return tokenizer(captions, padding="max_length", max_length=MAX_LENGTH).input_ids


def preprocess_fn(
    idx,
    feature_extractor,
    tokenizer,
    examples,
):
    model_inputs = {}
    captions = [cap[idx] for cap in examples["caption"]]
    model_inputs["labels"] = tokenization_fn(tokenizer, captions)

    model_inputs["pixel_values"] = feature_extractor(
        images=examples["image"], return_tensors="pt"
    )["pixel_values"]
    return model_inputs


def get_dataset(feature_extractor_model, text_decoder_model):
    if os.path.exists(CACHED_DS):
        return load_from_disk(CACHED_DS)

    ds = load_dataset("nlphuji/flickr30k", split="test")

    # splitting in train (80%), test (10%), validation (10%)
    ds = ds.train_test_split(test_size=0.2)
    test_and_eval = ds["test"].train_test_split(test_size=0.5)
    ds["test"] = test_and_eval["test"]
    ds["validation"] = test_and_eval["train"]

    feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_model)
    tokenizer = AutoTokenizer.from_pretrained(text_decoder_model)
    tokenizer.pad_token = tokenizer.eos_token

    for idx in range(5):
        mapped = ds.map(
            function=partial(preprocess_fn, idx, feature_extractor, tokenizer),
            batched=True,
            remove_columns=ds["train"].column_names,
        )
        if idx == 0:
            ds = mapped
        else:
            ds = concatenate_datasets([ds, mapped])

    ds.save_to_disk(CACHED_DS)
    return ds
