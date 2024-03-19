# distilvit

Same as https://huggingface.co/nlpconnect/vit-gpt2-image-captioning but with Distil-GPT-2

The train script is inspired from https://ankur3107.github.io/blogs/the-illustrated-image-captioning-using-transformers/#references

To install, use your favorite tools or you can run this:

```
python -m venv .
bin/pip install -r requirements.txt
bin/pip install -e .
```

To train against the 270k+ image & caption pairs (COCO and Flickr30k), make sure you have 2T of disk space, and run:

```
bin/train --dataset both
```

Once trained, you can try it out with the test script, to compare its output with `vit-gpt2-image-captioning`:

```
bin/python distilvit/infere.py
```
