# Copyright (c) ModelScope Contributors. All rights reserved.
"""ST LoRA over HTTP: train, read embeddings, save, recreate, resume and release."""

import argparse

import json
import numpy as np
from peft import LoraConfig
from transformers import AutoTokenizer

from twinkle import init_twinkle_client

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEXTS = [
    "A cat sits on a mat.",
    "There is a cat on the mat.",
    "A plane is flying.",
    "The dog runs in a park.",
    "A dog is running outdoors.",
    "The stock market closed.",
]


def encoded_batch(tokenizer):
    rows = [dict(tokenizer(text, truncation=True, max_length=128)) for text in TEXTS]
    for index, row in enumerate(rows):
        row["labels"] = [int(index % 3 == 0)]
    return rows


def configure(model, *, lr=1e-3, max_length=128):
    model.set_template("Template", max_length=max_length)
    model.set_processor("InputProcessor", padding_side="right")
    model.set_loss("InfonceLoss", temperature=0.1)
    model.set_optimizer("AdamW", lr=lr)


def vectors(model, batch):
    return np.asarray(model.forward_only(inputs=batch).result["embeddings"], dtype=float)


def step(model, batch):
    model.forward_backward(inputs=batch, task="embedding")
    model.clip_grad_and_step()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default=MODEL, help="Public model name in the server YAML")
    parser.add_argument("--tokenizer", default=MODEL, help="Original ST model or local cache directory")
    parser.add_argument("--api-key", default="st-example")
    args = parser.parse_args()
    batch = encoded_batch(AutoTokenizer.from_pretrained(args.tokenizer))
    client = init_twinkle_client(base_url=args.base_url, api_key=args.api_key)
    active = None
    try:
        active = client.model(args.model)
        active.add_adapter_to_model("first", LoraConfig(r=4, target_modules=["query", "value"]))
        configure(active)
        before = vectors(active, batch)
        for _ in range(2):
            step(active, batch)
        trained = vectors(active, batch)
        delta = float(np.max(np.abs(trained - before)))
        assert delta > 1e-6, "Training did not change embeddings"
        saved = active.save("st-example", save_optimizer=True, consumed_train_samples=2 * len(batch))
        active.remove_adapter()
        active = client.model(args.model)
        active.add_adapter_to_model("restored", LoraConfig(r=4, target_modules=["query", "value"]))
        configure(active)
        progress = active.resume_from_checkpoint(saved.twinkle_path)
        np.testing.assert_allclose(vectors(active, batch), trained, rtol=0, atol=2e-5)
        assert progress["consumed_train_samples"] == 2 * len(batch)
        step(active, batch)
        print(json.dumps({"checkpoint": saved.twinkle_path, "progress": progress, "embedding_delta": delta}))  # noqa: T201
    finally:
        try:
            if active is not None:
                active.remove_adapter()
        finally:
            client.close()


if __name__ == "__main__":
    main()
