#!/usr/bin/env python3
"""Verify the real cached MiniLM encoder on synthetic text without networking."""
import json
import os
import socket
from unittest.mock import patch

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
ENCODER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def main():
    with patch.object(socket.socket, "connect", side_effect=RuntimeError("Offline encoder check forbids networking")), \
         patch.object(socket, "create_connection", side_effect=RuntimeError("Offline encoder check forbids networking")):
        import numpy as np
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2", revision=ENCODER_REVISION,
                                    device="cpu", local_files_only=True)
        texts = ["A wholly synthetic patient enjoys walking outdoors.",
                 "A fictional person likes a walk in the park.",
                 "The synthetic spaceship flies between planets."]
        vectors = model.encode(texts, normalize_embeddings=True)
        similarities = vectors @ vectors.T
        assert vectors.shape == (3, 384), vectors.shape
        assert np.isfinite(vectors).all(), "Encoder returned nonfinite values"
        assert similarities[0, 1] > similarities[0, 2], "Related texts did not rank first"
        print(json.dumps({"shape": list(vectors.shape), "finite": bool(np.isfinite(vectors).all()),
                          "related_similarity": float(similarities[0, 1]),
                          "unrelated_similarity": float(similarities[0, 2]),
                          "device": "cpu", "revision": ENCODER_REVISION,
                          "network": "disabled", "passed": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
