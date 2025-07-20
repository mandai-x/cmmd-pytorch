# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Embedding models used in the CMMD calculation."""

from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import torch
import numpy as np

_CLIP_MODEL_NAME = "openai/clip-vit-large-patch14-336"


def get_device():
    """Get the best available device with preference: CUDA -> MPS -> CPU.
    
    Returns:
        torch.device: The best available device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def _resize_bicubic(images, size):
    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    images = torch.nn.functional.interpolate(images, size=(size, size), mode="bicubic")
    images = images.permute(0, 2, 3, 1).numpy()
    return images


class ClipEmbeddingModel:
    """CLIP image embedding calculator."""

    def __init__(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(_CLIP_MODEL_NAME)
        self.device = get_device()

        self._model = CLIPVisionModelWithProjection.from_pretrained(_CLIP_MODEL_NAME).eval()
        self._model = self._model.to(self.device)

        self.input_image_size = self.image_processor.crop_size["height"]

    @torch.no_grad()
    def embed(self, images, batch_size=4):
        """Computes CLIP embeddings for the given images.

        Args:
          images: An image array of shape (batch_size, height, width, 3). Values are
            in range [0, 1].
          batch_size: Number of images to process at once to control GPU memory usage.

        Returns:
          Embedding array of shape (batch_size, embedding_width).
        """
        
        # Process images in batches
        all_embeddings = []
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            
            batch_images = _resize_bicubic(batch_images, self.input_image_size)
            inputs = self.image_processor(
                images=batch_images,
                do_normalize=True,
                do_center_crop=False,
                do_resize=False,
                do_rescale=False,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            batch_embs = self._model(**inputs).image_embeds.cpu()
            batch_embs /= torch.linalg.norm(batch_embs, axis=-1, keepdims=True)
            all_embeddings.append(batch_embs)
        
        # Concatenate all batch embeddings
        image_embs = torch.cat(all_embeddings, dim=0)
        return image_embs
