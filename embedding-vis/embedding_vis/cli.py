from __future__ import annotations

import argparse
from pathlib import Path

from .visualizer import VisualizerConfig, build_visualization


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build interactive 2D or 3D embedding visualizations with multiple reduction methods."
    )
    subparsers = parser.add_subparsers(dest="reduction_method", required=True)

    umap_parser = subparsers.add_parser(
        "umap",
        help="Reduce embeddings with UMAP before visualization.",
        description="Build an interactive 2D or 3D embedding visualization using UMAP.",
    )
    add_shared_arguments(umap_parser)
    umap_parser.add_argument(
        "--neighbors",
        type=int,
        default=30,
        help="UMAP n_neighbors parameter.",
    )
    umap_parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist parameter.",
    )

    mds_parser = subparsers.add_parser(
        "mds",
        help="Reduce embeddings with metric MDS before visualization.",
        description="Build an interactive 2D or 3D embedding visualization using metric MDS.",
    )
    add_shared_arguments(mds_parser)
    mds_parser.add_argument(
        "--mds-max-iter",
        type=int,
        default=300,
        help="Maximum optimization iterations for metric MDS stress minimization.",
    )
    mds_parser.add_argument(
        "--mds-lr",
        type=float,
        default=0.05,
        help="Learning rate for metric MDS stress optimization.",
    )
    mds_parser.add_argument(
        "--mds-tol",
        type=float,
        default=1e-7,
        help="Early-stop tolerance on stress improvement between iterations.",
    )

    return parser


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        default="bert-base-uncased",
        help="Hugging Face model name or local path used to load the input embedding layer.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("embedding_vis.html"),
        help="Path for the generated Plotly HTML file.",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        choices=(2, 3),
        default=3,
        help="Projection and rendering dimensionality: 2 for a planar scatter, 3 for the interactive 3D view.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=512,
        help="Maximum number of vocabulary embedding rows to visualize from the model.",
    )
    parser.add_argument(
        "--token-regex",
        default=None,
        help="Optional regex filter applied to decoded token text before sampling and reduction.",
    )
    parser.add_argument(
        "--sampling-mode",
        choices=("top", "random"),
        default="top",
        help="How to choose model embedding rows before reduction: 'top' keeps the first rows, 'random' draws a reproducible random sample using --seed.",
    )
    parser.add_argument(
        "--extra-vectors",
        type=Path,
        default=None,
        help="Optional .pt file containing additional vectors to append before reduction.",
    )
    parser.add_argument(
        "--extra-labels-key",
        default="texts",
        help="Dictionary key for hover text labels inside the extra .pt file.",
    )
    parser.add_argument(
        "--extra-vectors-key",
        default="vectors",
        help="Dictionary key for vector data inside the extra .pt file.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for the MDS reducer and tensor loading. Use 'auto' to prefer cuda, then mps, then cpu.",
    )
    parser.add_argument(
        "--metric",
        default="cosine",
        help="Distance metric used by the selected reduction method.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible reduction output.",
    )
    parser.add_argument(
        "--radius-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to the inferred point radius.",
    )
    parser.add_argument(
        "--initial-zoom",
        type=float,
        default=6.0,
        help="Initial camera distance hint for the 3D embedding cloud.",
    )
    parser.add_argument(
        "--min-zoom",
        type=float,
        default=-2.0,
        help="Minimum allowed zoom level.",
    )
    parser.add_argument(
        "--max-zoom",
        type=float,
        default=18.0,
        help="Maximum allowed zoom level.",
    )


def main() -> None:
    args = build_parser().parse_args()
    config = VisualizerConfig(
        reduction_method=args.reduction_method,
        model_name=args.model,
        output_path=args.output,
        dimensions=args.dimensions,
        max_points=args.max_points,
        token_filter_regex=args.token_regex,
        sampling_mode=args.sampling_mode,
        extra_vectors_path=args.extra_vectors,
        extra_labels_key=args.extra_labels_key,
        extra_vectors_key=args.extra_vectors_key,
        device=args.device,
        umap_neighbors=getattr(args, "neighbors", 30),
        umap_min_dist=getattr(args, "min_dist", 0.1),
        reduction_metric=args.metric,
        random_seed=args.seed,
        mds_max_iter=getattr(args, "mds_max_iter", 300),
        mds_learning_rate=getattr(args, "mds_lr", 0.05),
        mds_tolerance=getattr(args, "mds_tol", 1e-7),
        radius_scale=args.radius_scale,
        initial_zoom=args.initial_zoom,
        min_zoom=args.min_zoom,
        max_zoom=args.max_zoom,
    )
    build_visualization(config)
