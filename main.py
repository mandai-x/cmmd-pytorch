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

"""The main entry point for the CMMD calculation."""

import click
import distance
import embedding
import io_util
import numpy as np


def compute_cmmd(ref_dir, eval_dir, ref_embed_file=None, batch_size=32, max_count=-1):
    """Calculates the CMMD distance between reference and eval image sets.

    Args:
      ref_dir: Path to the directory containing reference images.
      eval_dir: Path to the directory containing images to be evaluated.
      ref_embed_file: Path to the pre-computed embedding file for the reference images.
      batch_size: Batch size used in the CLIP embedding calculation.
      max_count: Maximum number of images to use from each directory. A
        non-positive value reads all images available except for the images
        dropped due to batching.

    Returns:
      The CMMD value between the image sets.
    """
    if ref_dir and ref_embed_file:
        raise ValueError("`ref_dir` and `ref_embed_file` both cannot be set at the same time.")
    embedding_model = embedding.ClipEmbeddingModel()
    if ref_embed_file is not None:
        ref_embs = np.load(ref_embed_file).astype("float32")
    else:
        ref_embs = io_util.compute_embeddings_for_dir(ref_dir, embedding_model, batch_size, max_count).astype(
            "float32"
        )
    eval_embs = io_util.compute_embeddings_for_dir(eval_dir, embedding_model, batch_size, max_count).astype("float32")
    val = distance.mmd(ref_embs, eval_embs)
    return val.numpy()


@click.command()
@click.argument('ref_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument('eval_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--batch-size', '-b', default=32, type=int, 
              help='Batch size for embedding generation (default: 32)')
@click.option('--max-count', '-m', default=-1, type=int, 
              help='Maximum number of images to read from each directory. Use -1 for all images (default: -1)')
@click.option('--ref-embed-file', '-r', type=click.Path(exists=True, file_okay=True, dir_okay=False), 
              help='Path to the pre-computed embedding file for the reference images')
def main(ref_dir, eval_dir, batch_size, max_count, ref_embed_file):
    """Calculate CLIP Maximum Mean Discrepancy (CMMD) between two image directories.
    
    This tool computes the CMMD distance between a reference set of images and an evaluation set of images.
    The CMMD metric is useful for evaluating the quality of generated images by comparing them to reference images.
    
    Arguments:
        ref_dir: Path to the directory containing reference images. These are typically high-quality,
                 real images that serve as the ground truth for comparison.
        eval_dir: Path to the directory containing images to be evaluated. These are typically
                  generated images whose quality is being assessed against the reference set.
    """
    try:
        cmmd_value = compute_cmmd(ref_dir, eval_dir, ref_embed_file, batch_size, max_count)
        click.echo(f"The CMMD value is: {cmmd_value:.3f}")
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    main()
