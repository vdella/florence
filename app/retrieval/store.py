from dataclasses import dataclass

import torch


@dataclass
class CachedDocument:
    chunks: list[str]
    embeddings: torch.Tensor


DOCUMENT_CACHE: dict[str, CachedDocument] = {}