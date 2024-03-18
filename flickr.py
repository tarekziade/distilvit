from datasets import load_dataset, load_from_disk
from functools import partial
import os
from transformers import (
    AutoTokenizer,
    AutoFeatureExtractor,
)


MAX_LENGTH = 128
CACHED_DS = os.path.join(os.path.dirname(__file__), "cache", "flickr30k")


def tokenization_fn(tokenizer, captions):
    return tokenizer(captions, padding="max_length", max_length=MAX_LENGTH).input_ids


def preprocess_fn(
    feature_extractor,
    tokenizer,
    examples,
):
    model_inputs = {}
    for i in range(5):
        captions = [cap[i] for cap in examples["caption"]]
        model_inputs[f"labels_{i}"] = tokenization_fn(tokenizer, captions)

    model_inputs["pixel_values"] = feature_extractor(
        images=examples["image"], return_tensors="pt"
    )["pixel_values"]
    return model_inputs


image_encoder_model = "google/vit-base-patch16-224-in21k"

ds = load_dataset("nlphuji/flickr30k")
ds = ds["test"].train_test_split(test_size=0.2)

text_decode_model = "distilbert/distilgpt2"
feature_extractor = AutoFeatureExtractor.from_pretrained(image_encoder_model)

tokenizer = AutoTokenizer.from_pretrained(text_decode_model)
tokenizer.pad_token = tokenizer.eos_token


ds = ds.map(
    function=partial(preprocess_fn, feature_extractor, tokenizer),
    batched=True,
    remove_columns=ds["train"].column_names,
)
# save the mapped dataset so we can reuse it
ds.save_to_disk(CACHED_DS)
