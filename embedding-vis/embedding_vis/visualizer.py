from __future__ import annotations

import importlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import umap
from transformers import AutoModel, AutoTokenizer


@dataclass
class VisualizerConfig:
    reduction_method: str
    model_name: str
    output_path: Path
    dimensions: int
    max_points: int
    token_filter_regex: str | None
    token_include_regex: str | None
    sampling_mode: str
    extra_vectors_path: Path | None
    extra_labels_key: str
    extra_vectors_key: str
    device: str
    umap_neighbors: int
    umap_min_dist: float
    reduction_metric: str
    random_seed: int
    mds_max_iter: int
    mds_learning_rate: float
    mds_tolerance: float
    radius_scale: float
    initial_zoom: float = 6.0
    min_zoom: float = -2.0
    max_zoom: float = 18.0
    apply_pca: bool = False
    mds_extra_weight: float = 1.0


def build_visualization(config: VisualizerConfig) -> Path:
    records = load_embedding_records(config)
    if not records:
        raise ValueError(
            "No vectors available after token filtering. "
            "Adjust --token-regex/--include or provide extra vectors."
        )
    vectors = np.asarray([record["vector"] for record in records], dtype=np.float32)

    point_weights = np.ones(len(records), dtype=np.float32)
    if config.mds_extra_weight != 1.0:
        for i, record in enumerate(records):
            if record["source"].startswith("extra:"):
                point_weights[i] = config.mds_extra_weight

    projected = reduce_embeddings(vectors, config, point_weights)
    dataframe = build_dataframe(records, projected)
    figure = create_figure(dataframe, config)
    plotted_positions = np.asarray(dataframe["position"].tolist(), dtype=np.float32)
    neighbor_payload = build_neighbor_panel_payload(vectors, plotted_positions, config)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    html = build_interactive_html(figure, neighbor_payload)
    config.output_path.write_text(html, encoding="utf-8")
    print(f"Wrote interactive visualization to {config.output_path}")
    return config.output_path


def load_embedding_records(config: VisualizerConfig) -> list[dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(config.model_name, trust_remote_code=True)
    embedding_layer = model.get_input_embeddings()
    if embedding_layer is None:
        raise ValueError(
            f"Model '{config.model_name}' does not expose an input embedding layer."
        )

    weights = embedding_layer.weight.detach().to(torch.float32).cpu()
    filtered_tokens = collect_filtered_tokens(tokenizer, weights.shape[0], config)
    limit = min(config.max_points, len(filtered_tokens))
    records: list[dict[str, Any]] = []

    selected_indices = select_token_ids(len(filtered_tokens), limit, config)
    selected_tokens: list[tuple[int, str]] = [
        filtered_tokens[filtered_index] for filtered_index in selected_indices
    ]

    include_tokens = collect_include_tokens(tokenizer, weights.shape[0], config)
    selected_token_ids = {token_id for token_id, _ in selected_tokens}
    forced_additions = 0
    for token_id, cleaned_text in include_tokens:
        if token_id in selected_token_ids:
            continue
        selected_tokens.append((token_id, cleaned_text))
        selected_token_ids.add(token_id)
        forced_additions += 1

    for token_id, cleaned_text in selected_tokens:
        records.append(
            {
                "text": cleaned_text,
                "label": cleaned_text,
                "source": f"model:{config.model_name}",
                "vector": weights[token_id].numpy(),
                "color": [66, 135, 245],
            }
        )

    print(
        "Final collected token number: "
        f"{len(selected_tokens)} (sampled={len(selected_indices)}, "
        f"forced={forced_additions}, regex-matched={len(filtered_tokens)})"
    )

    if config.extra_vectors_path is not None:
        model_vecs = np.asarray(
            [record["vector"] for record in records], dtype=np.float32
        )
        model_txts = [record["text"] for record in records]
        records.extend(
            load_extra_vector_records(config, weights.shape[1], model_vecs, model_txts)
        )

    return records


def collect_filtered_tokens(
    tokenizer: Any, total_points: int, config: VisualizerConfig
) -> list[tuple[int, str]]:
    matcher = build_regex_matcher(config.token_filter_regex, "--token-regex")
    filtered_tokens: list[tuple[int, str]] = []
    for token_id in range(total_points):
        token_text = decode_token_text(tokenizer, token_id)
        cleaned_text = sanitize_token_text(token_text)
        if matcher is not None and not matcher.search(cleaned_text):
            continue
        filtered_tokens.append((token_id, cleaned_text))
    return filtered_tokens


def collect_include_tokens(
    tokenizer: Any, total_points: int, config: VisualizerConfig
) -> list[tuple[int, str]]:
    matcher = build_regex_matcher(config.token_include_regex, "--include")
    if matcher is None:
        return []

    include_tokens: list[tuple[int, str]] = []
    for token_id in range(total_points):
        token_text = decode_token_text(tokenizer, token_id)
        cleaned_text = sanitize_token_text(token_text)
        if matcher.search(cleaned_text):
            include_tokens.append((token_id, cleaned_text))
    return include_tokens


def build_regex_matcher(
    pattern: str | None, argument_name: str
) -> re.Pattern[str] | None:
    if pattern is None:
        return None
    try:
        return re.compile(pattern)
    except re.error as error:
        raise ValueError(
            f"Invalid {argument_name} pattern: {pattern!r}. {error}"
        ) from error


def select_token_ids(
    total_points: int, limit: int, config: VisualizerConfig
) -> list[int]:
    if limit <= 0:
        return []
    if config.sampling_mode == "top":
        return list(range(limit))
    if config.sampling_mode == "random":
        generator = np.random.default_rng(config.random_seed)
        return generator.choice(total_points, size=limit, replace=False).tolist()
    raise ValueError(f"Unsupported sampling mode: {config.sampling_mode}")


def sanitize_token_text(token_text: str | None) -> str:
    if token_text is None:
        return "<unknown-token>"
    sanitized = token_text.replace("Ġ", "▁").replace("Ċ", "\\n")
    return sanitized if sanitized else "<empty-token>"


def decode_token_text(tokenizer: Any, token_id: int) -> str:
    decoded = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
    if decoded:
        return decoded

    raw_token = tokenizer.convert_ids_to_tokens(token_id)
    return "" if raw_token is None else str(raw_token)


def load_extra_vector_records(
    config: VisualizerConfig,
    expected_width: int,
    model_vectors: np.ndarray | None = None,
    model_texts: list[str] | None = None,
) -> list[dict[str, Any]]:
    if config.extra_vectors_path is None:
        return []

    payload = torch.load(
        config.extra_vectors_path, map_location=resolve_tensor_load_device(config)
    )
    vectors, texts, source_key = parse_extra_payload(payload, config)

    if vectors.shape[1] != expected_width:
        raise ValueError(
            "Extra vectors must have the same embedding width as the model input embeddings: "
            f"expected {expected_width}, got {vectors.shape[1]}."
        )
    if len(texts) != vectors.shape[0]:
        raise ValueError(
            f"Extra vector labels count ({len(texts)}) does not match vector rows ({vectors.shape[0]})."
        )

    if model_vectors is not None and model_texts is not None:
        model_vectors_norm = model_vectors / np.linalg.norm(
            model_vectors, axis=1, keepdims=True
        ).clip(min=1e-12)
    else:
        model_vectors_norm = None

    records: list[dict[str, Any]] = []
    for index, text in enumerate(texts):
        label = str(text)
        if model_vectors_norm is not None:
            sims = model_vectors_norm @ (vectors[index] / max(np.linalg.norm(vectors[index]), 1e-12))
            nearest_indices = np.argsort(-sims)[:3]
            nearest_labels = [model_texts[i] for i in nearest_indices]
            label = f"[{index}] {text}   ❮ {', '.join(nearest_labels)}"
        else:
            label = f"[{index}] {text}"
        records.append(
            {
                "text": label,
                "label": label,
                "source": f"extra:{config.extra_vectors_path.name}",
                "vector": vectors[index].astype(np.float32, copy=False),
                "color": [245, 158, 11],
                "weight": 5.0,
            }
        )
    return records


def parse_extra_payload(
    payload: Any, config: VisualizerConfig
) -> tuple[np.ndarray, list[str], str]:
    """Returns (vectors, labels, source_key)."""
    raw_vectors: Any = None
    raw_labels: Any = None
    source_key: str = ""

    if isinstance(payload, dict):
        if config.extra_vectors_key in payload:
            raw_vectors = payload[config.extra_vectors_key]
            raw_labels = payload.get(config.extra_labels_key, None)
            source_key = config.extra_vectors_key
        else:
            tensor_candidates = {
                k: v
                for k, v in payload.items()
                if isinstance(v, (torch.Tensor, np.ndarray))
            }
            if not tensor_candidates:
                raise ValueError(
                    f"Extra vector file is missing '{config.extra_vectors_key}' "
                    f"and no tensor values found. Available keys: {list(payload.keys())}"
                )
            best_key = list(tensor_candidates.keys())[0]
            raw_vectors = tensor_candidates[best_key]
            source_key = best_key
    elif isinstance(payload, torch.Tensor):
        raw_vectors = payload
        source_key = "tensor"
    else:
        raise ValueError(
            "Unsupported extra vector payload. Expected either a tensor or a dictionary."
        )

    if hasattr(raw_vectors, "ndim") and raw_vectors.ndim == 3:
        if raw_vectors.shape[0] == 1:
            raw_vectors = raw_vectors.squeeze(0)
        elif raw_vectors.shape[1] == 1:
            raw_vectors = raw_vectors.squeeze(1)
        else:
            raise ValueError(
                f"Extra vectors have 3 dimensions {tuple(raw_vectors.shape)}. "
                "Cannot auto-squeeze; expected shape (N, D) or (1, N, D)."
            )

    vectors = to_numpy_matrix(raw_vectors)

    if vectors.ndim != 2:
        raise ValueError(
            f"Extra vectors must be a 2D tensor/array, received shape {tuple(vectors.shape)}."
        )

    if raw_labels is not None:
        raw_labels_list = list(raw_labels)
    else:
        raw_labels_list = [f"{source_key}[{i}]" for i in range(vectors.shape[0])]

    labels = [str(item) for item in raw_labels_list]
    return vectors, labels, source_key


def to_numpy_matrix(raw_vectors: Any) -> np.ndarray:
    if isinstance(raw_vectors, torch.Tensor):
        return raw_vectors.detach().to(torch.float32).cpu().numpy()
    if isinstance(raw_vectors, np.ndarray):
        return raw_vectors.astype(np.float32, copy=False)
    return np.asarray(raw_vectors, dtype=np.float32)


def reduce_embeddings(
    vectors: np.ndarray, config: VisualizerConfig, point_weights: np.ndarray | None = None
) -> np.ndarray:
    processed = maybe_apply_pca(vectors, config)
    if config.reduction_method == "umap":
        return reduce_with_umap(processed, config)
    if config.reduction_method == "mds":
        return reduce_with_mds(processed, config, point_weights)
    raise ValueError(f"Unsupported reduction method: {config.reduction_method}")


def reduce_with_umap(vectors: np.ndarray, config: VisualizerConfig) -> np.ndarray:
    reducer = umap.UMAP(
        n_components=config.dimensions,
        n_neighbors=config.umap_neighbors,
        min_dist=config.umap_min_dist,
        metric=config.reduction_metric,
        random_state=config.random_seed,
    )
    reduced = reducer.fit_transform(vectors)
    return np.asarray(reduced, dtype=np.float32)


def reduce_with_mds(
    vectors: np.ndarray,
    config: VisualizerConfig,
    point_weights: np.ndarray | None = None,
) -> np.ndarray:
    mds_module = importlib.import_module("embedding_vis.mds")
    return mds_module.reduce_with_mds(
        vectors,
        dimensions=config.dimensions,
        metric=config.reduction_metric,
        device_preference=config.device,
        random_seed=config.random_seed,
        max_iter=config.mds_max_iter,
        learning_rate=config.mds_learning_rate,
        tolerance=config.mds_tolerance,
        point_weights=point_weights,
    )


def resolve_tensor_load_device(config: VisualizerConfig) -> str:
    if config.device == "auto":
        return "cpu"
    return config.device


def maybe_apply_pca(vectors: np.ndarray, config: VisualizerConfig) -> np.ndarray:
    if config.apply_pca:
        raise NotImplementedError(
            "PCA preprocessing is intentionally left as an extension point and is not enabled yet."
        )
    return vectors


def build_dataframe(
    records: list[dict[str, Any]], projected: np.ndarray
) -> pd.DataFrame:
    max_abs = float(np.max(np.abs(projected))) if projected.size else 1.0
    safe_scale = max(max_abs, 1e-6)
    positions = projected / safe_scale
    radii = infer_point_radii(positions)
    for i, record in enumerate(records):
        radii[i] *= record.get("weight", 1.0)
    dataframe = pd.DataFrame(
        {
            "point_id": list(range(len(records))),
            "position": positions.tolist(),
            "text": [record["text"] for record in records],
            "label": [record["label"] for record in records],
            "source": [record["source"] for record in records],
            "color": [record["color"] for record in records],
            "radius": radii.tolist(),
        }
    )
    return dataframe


def create_figure(dataframe: pd.DataFrame, config: VisualizerConfig) -> go.Figure:
    dataframe = dataframe.copy()
    dataframe["radius"] = dataframe["radius"] * config.radius_scale
    positions = np.asarray(dataframe["position"].tolist(), dtype=np.float32)
    marker_sizes = scale_marker_sizes(dataframe["radius"].to_numpy(dtype=np.float32))
    colors = [to_plotly_color(color) for color in dataframe["color"]]
    hover_text = [
        f"<b>{label}</b><br>{source}"
        for label, source in zip(dataframe["label"], dataframe["source"])
    ]
    point_ids = dataframe["point_id"].to_list()
    point_text = dataframe["text"].tolist()
    point_sources = dataframe["source"].tolist()

    if config.dimensions == 2:
        return create_2d_figure(
            positions,
            marker_sizes,
            colors,
            hover_text,
            point_text,
            point_sources,
            point_ids,
        )

    return create_3d_figure(
        positions,
        marker_sizes,
        colors,
        hover_text,
        point_text,
        point_sources,
        point_ids,
        config,
    )


def create_2d_figure(
    positions: np.ndarray,
    marker_sizes: np.ndarray,
    colors: list[str],
    hover_text: list[str],
    point_text: list[str],
    point_sources: list[str],
    point_ids: list[int],
) -> go.Figure:
    scatter = go.Scatter(
        x=positions[:, 0],
        y=positions[:, 1],
        mode="markers",
        text=point_text,
        customdata=build_customdata(point_ids, point_text, point_sources),
        hovertext=hover_text,
        hovertemplate="%{hovertext}<extra></extra>",
        marker={
            "size": marker_sizes,
            "color": colors,
            "opacity": 0.85,
            "line": {"width": 0},
        },
    )
    figure = go.Figure(data=[scatter])
    figure.update_layout(
        template="plotly_dark",
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        dragmode="pan",
        uirevision="token-filter",
        xaxis=build_xy_axis_config("X"),
        yaxis=build_xy_axis_config("Y"),
        showlegend=False,
    )
    figure.update_yaxes(scaleanchor="x", scaleratio=1)
    return figure


def create_3d_figure(
    positions: np.ndarray,
    marker_sizes: np.ndarray,
    colors: list[str],
    hover_text: list[str],
    point_text: list[str],
    point_sources: list[str],
    point_ids: list[int],
    config: VisualizerConfig,
) -> go.Figure:
    scatter = go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode="markers",
        text=point_text,
        customdata=build_customdata(point_ids, point_text, point_sources),
        hovertext=hover_text,
        hovertemplate="%{hovertext}<extra></extra>",
        marker={
            "size": marker_sizes,
            "color": colors,
            "opacity": 0.85,
            "line": {"width": 0},
        },
    )
    camera = infer_camera(positions, config)
    figure = go.Figure(data=[scatter])
    figure.update_layout(
        template="plotly_dark",
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="#111827",
        scene_dragmode="pan",
        uirevision="token-filter",
        scene={
            "xaxis": build_axis_config("X"),
            "yaxis": build_axis_config("Y"),
            "zaxis": build_axis_config("Z"),
            "aspectmode": "data",
            "camera": camera,
        },
        showlegend=False,
    )
    figure.update_scenes(camera_projection_type="perspective")
    return figure


def infer_point_radii(positions: np.ndarray) -> np.ndarray:
    if len(positions) == 0:
        return np.asarray([], dtype=np.float32)
    if len(positions) == 1:
        return np.asarray([0.02], dtype=np.float32)

    deltas = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest_distances = np.min(distances, axis=1)

    finite_nearest = nearest_distances[np.isfinite(nearest_distances)]
    if finite_nearest.size == 0:
        return np.full(len(positions), 0.02, dtype=np.float32)

    baseline = float(np.median(finite_nearest))
    baseline = max(baseline, 1e-4)
    clipped = np.clip(nearest_distances, baseline * 0.35, baseline * 2.5)
    radii = clipped * 0.18
    return np.clip(radii, 0.004, 0.03).astype(np.float32)


def infer_target(dataframe: pd.DataFrame) -> list[float]:
    positions = np.asarray(dataframe["position"].tolist(), dtype=np.float32)
    if positions.size == 0:
        return [0.0, 0.0, 0.0]
    center = positions.mean(axis=0)
    return [float(center[0]), float(center[1]), float(center[2])]


def scale_marker_sizes(radii: np.ndarray) -> np.ndarray:
    if radii.size == 0:
        return np.asarray([], dtype=np.float32)
    min_radius = float(np.min(radii))
    max_radius = float(np.max(radii))
    if np.isclose(min_radius, max_radius):
        return np.full(radii.shape, 5.0, dtype=np.float32)
    normalized = (radii - min_radius) / (max_radius - min_radius)
    return (normalized * 6.0 + 3.0).astype(np.float32)


def to_plotly_color(color: list[int]) -> str:
    red, green, blue = color
    return f"rgb({red},{green},{blue})"


def infer_camera(
    positions: np.ndarray, config: VisualizerConfig
) -> dict[str, dict[str, float]]:
    if positions.size == 0:
        return {
            "eye": {"x": 1.4, "y": 1.4, "z": 1.1},
            "center": {"x": 0.0, "y": 0.0, "z": 0.0},
            "up": {"x": 0.0, "y": 0.0, "z": 1.0},
        }

    center = positions.mean(axis=0)
    spread = np.ptp(positions, axis=0)
    radius = max(float(np.linalg.norm(spread)), 0.25)
    zoom_factor = 1.0 / max(config.initial_zoom, 0.5)
    distance = max(radius * (1.2 + zoom_factor * 3.0), 0.6)
    return {
        "eye": {
            "x": float(center[0] + distance),
            "y": float(center[1] + distance),
            "z": float(center[2] + distance * 0.75),
        },
        "center": {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])},
        "up": {"x": 0.0, "y": 0.0, "z": 1.0},
    }


def build_axis_config(title: str) -> dict[str, Any]:
    return {
        "visible": True,
        "title": {"text": title, "font": {"color": "#F9FAFB", "size": 12}},
        "showbackground": True,
        "backgroundcolor": "rgba(17, 24, 39, 0.35)",
        "gridcolor": "rgba(148, 163, 184, 0.25)",
        "zerolinecolor": "rgba(148, 163, 184, 0.55)",
        "showspikes": False,
        "ticks": "outside",
        "tickfont": {"color": "#E5E7EB", "size": 10},
    }


def build_xy_axis_config(title: str) -> dict[str, Any]:
    return {
        "title": {"text": title, "font": {"color": "#F9FAFB", "size": 12}},
        "showgrid": True,
        "gridcolor": "rgba(148, 163, 184, 0.25)",
        "zeroline": True,
        "zerolinecolor": "rgba(148, 163, 184, 0.55)",
        "showline": True,
        "linecolor": "rgba(148, 163, 184, 0.55)",
        "ticks": "outside",
        "tickfont": {"color": "#E5E7EB", "size": 10},
    }


def build_customdata(
    point_ids: list[int], point_text: list[str], point_sources: list[str]
) -> list[list[str | int]]:
    return [
        [point_id, text, source]
        for point_id, text, source in zip(point_ids, point_text, point_sources)
    ]


def build_neighbor_panel_payload(
    vectors: np.ndarray, plotted_positions: np.ndarray, config: VisualizerConfig
) -> dict[str, Any]:
    point_count = len(vectors)
    max_neighbors = min(100, max(point_count - 1, 0))
    default_neighbors = min(20, max_neighbors)
    return {
        "default_space": "original",
        "spaces": {
            "original": build_neighbor_space_payload(
                vectors,
                label="Original Space",
                description="Nearest neighbors in the full embedding space before reduction.",
                metric=resolve_neighbor_metric(config.reduction_metric),
                max_neighbors=max_neighbors,
                default_neighbors=default_neighbors,
                device_preference=config.device,
            ),
            "projected": build_neighbor_space_payload(
                plotted_positions,
                label="Projected Space",
                description="Nearest neighbors in the displayed 2D/3D plot coordinates.",
                metric="euclidean",
                max_neighbors=max_neighbors,
                default_neighbors=default_neighbors,
                device_preference=config.device,
            ),
        },
    }


def build_neighbor_space_payload(
    points: np.ndarray,
    *,
    label: str,
    description: str,
    metric: str,
    max_neighbors: int,
    default_neighbors: int,
    device_preference: str,
) -> dict[str, Any]:
    if max_neighbors == 0 or len(points) == 0:
        return {
            "label": label,
            "description": description,
            "metric": metric,
            "max_neighbors": 0,
            "default_neighbors": 0,
            "neighbors": {},
        }

    indices, distances = compute_top_neighbors(
        points,
        metric=metric,
        max_neighbors=max_neighbors,
        device_preference=device_preference,
    )
    neighbors = {
        str(point_id): [
            [int(neighbor_id), float(distance)]
            for neighbor_id, distance in zip(point_indices, point_distances)
        ]
        for point_id, point_indices, point_distances in zip(
            range(len(points)), indices, distances
        )
    }
    return {
        "label": label,
        "description": description,
        "metric": metric,
        "max_neighbors": max_neighbors,
        "default_neighbors": default_neighbors,
        "neighbors": neighbors,
    }


def resolve_neighbor_metric(metric: str) -> str:
    supported_metrics = {
        "cosine",
        "euclidean",
        "l2",
        "sqeuclidean",
        "squared_euclidean",
        "manhattan",
        "cityblock",
        "l1",
    }
    normalized_metric = metric.lower()
    if normalized_metric in supported_metrics:
        return normalized_metric

    print(
        "Nearest-points panel metric "
        f"{metric!r} is not supported; falling back to 'cosine'."
    )
    return "cosine"


def compute_top_neighbors(
    vectors: np.ndarray,
    *,
    metric: str,
    max_neighbors: int,
    device_preference: str,
) -> tuple[np.ndarray, np.ndarray]:
    device = resolve_neighbor_device(device_preference)
    try:
        return compute_top_neighbors_torch(vectors, metric, max_neighbors, device)
    except (RuntimeError, NotImplementedError) as error:
        if device_preference == "auto" and device.type != "cpu":
            print(
                "Falling back to CPU for the nearest-points panel because "
                f"{device.type} failed: {error}"
            )
            return compute_top_neighbors_torch(
                vectors, metric, max_neighbors, torch.device("cpu")
            )
        raise


def compute_top_neighbors_torch(
    vectors: np.ndarray,
    metric: str,
    max_neighbors: int,
    device: torch.device,
    chunk_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    points = torch.as_tensor(vectors, dtype=resolve_neighbor_dtype(device), device=device)
    point_count = points.shape[0]
    if point_count == 0 or max_neighbors <= 0:
        empty = np.zeros((point_count, 0), dtype=np.int64)
        return empty, empty.astype(np.float32)

    neighbor_count = min(max_neighbors, point_count - 1)
    all_indices: list[torch.Tensor] = []
    all_distances: list[torch.Tensor] = []
    normalized_metric = metric.lower()

    normalized_points: torch.Tensor | None = None
    if normalized_metric == "cosine":
        norms = torch.linalg.norm(points, dim=1, keepdim=True).clamp_min(1e-12)
        normalized_points = points / norms

    for start in range(0, point_count, chunk_size):
        end = min(start + chunk_size, point_count)
        query_rows = torch.arange(end - start, device=device)
        query_columns = torch.arange(start, end, device=device)

        if normalized_metric in {"euclidean", "l2"}:
            distances = torch.cdist(points[start:end], points, p=2.0)
        elif normalized_metric in {"sqeuclidean", "squared_euclidean"}:
            distances = torch.cdist(points[start:end], points, p=2.0).square()
        elif normalized_metric in {"manhattan", "cityblock", "l1"}:
            distances = torch.cdist(points[start:end], points, p=1.0)
        elif normalized_metric == "cosine":
            if normalized_points is None:
                raise RuntimeError("Normalized points were not initialized for cosine distance.")
            similarities = normalized_points[start:end] @ normalized_points.T
            distances = 1.0 - similarities.clamp(-1.0, 1.0)
        else:
            raise ValueError(
                "Unsupported nearest-neighbor metric: "
                f"{metric}. Supported metrics are cosine, euclidean, sqeuclidean, manhattan, cityblock, and l1."
            )

        distances[query_rows, query_columns] = torch.inf
        values, indices = torch.topk(
            distances, k=neighbor_count, dim=1, largest=False
        )
        all_indices.append(indices.cpu())
        all_distances.append(values.cpu())

    neighbor_indices = torch.cat(all_indices, dim=0).numpy().astype(np.int64, copy=False)
    neighbor_distances = torch.cat(all_distances, dim=0).numpy().astype(
        np.float32, copy=False
    )
    return neighbor_indices, neighbor_distances


def resolve_neighbor_device(device_preference: str) -> torch.device:
    requested = device_preference.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_preference)


def resolve_neighbor_dtype(device: torch.device) -> torch.dtype:
    if device.type in {"cuda", "mps"}:
        return torch.float32
    return torch.float64


def build_interactive_html(figure: go.Figure, neighbor_payload: dict[str, Any]) -> str:
    plot_html = figure.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={
            "displayModeBar": True,
            "scrollZoom": True,
        },
        div_id="embedding-vis-plot",
    )
    neighbor_payload_json = serialize_json_for_script(neighbor_payload)
    default_space = str(neighbor_payload.get("default_space", "original"))
    default_payload = neighbor_payload["spaces"][default_space]
    neighbor_slider_max = max(1, int(default_payload["max_neighbors"]))
    neighbor_slider_value = max(1, int(default_payload["default_neighbors"]))
    neighbor_slider_disabled = "disabled" if default_payload["max_neighbors"] == 0 else ""
    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>embedding-vis</title>
  <style>
    :root {{
      color-scheme: dark;
      font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif;
      background: #111827;
      color: #f9fafb;
    }}
    html, body {{
      height: 100%;
    }}
    body {{
      margin: 0;
      background: #111827;
      color: #f9fafb;
    }}
    .app-shell {{
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow: hidden;
    }}
    #content-shell {{
      flex: 1;
      min-height: 0;
      min-width: 0;
      display: flex;
    }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: end;
      flex: 0 0 auto;
      padding: 16px 20px 12px;
      border-bottom: 1px solid rgba(148, 163, 184, 0.24);
      background: rgba(17, 24, 39, 0.96);
      z-index: 10;
      backdrop-filter: blur(12px);
    }}
    .field {{
      display: flex;
      flex-direction: column;
      gap: 6px;
      min-width: 220px;
    }}
    .field label, .checkbox {{
      font-size: 12px;
      color: #cbd5e1;
    }}
    .checkbox {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-height: 40px;
    }}
    .toolbar input[type=\"text\"] {{
      border: 1px solid rgba(148, 163, 184, 0.35);
      border-radius: 10px;
      background: #0f172a;
      color: #f8fafc;
      padding: 10px 12px;
      font-size: 14px;
      outline: none;
    }}
    .toolbar input[type=\"text\"]:focus {{
      border-color: #60a5fa;
      box-shadow: 0 0 0 3px rgba(96, 165, 250, 0.18);
    }}
    .toolbar button {{
      border: 0;
      border-radius: 10px;
      padding: 10px 14px;
      background: #2563eb;
      color: white;
      font-size: 14px;
      cursor: pointer;
    }}
    .toolbar button:hover {{
      background: #1d4ed8;
    }}
    .status {{
      display: flex;
      flex-direction: column;
      gap: 4px;
      color: #cbd5e1;
      font-size: 13px;
      margin-left: auto;
      min-width: 220px;
    }}
    .status strong {{
      color: #f9fafb;
      font-size: 14px;
    }}
    .status-error {{
      color: #fca5a5;
      min-height: 18px;
    }}
    #plot-container {{
      flex: 1;
      min-height: 0;
      min-width: 0;
      display: flex;
    }}
    #plot-container > div {{
      flex: 1;
      min-height: 0;
      min-width: 0;
    }}
    #embedding-vis-plot {{
      width: 100%;
      height: 100% !important;
    }}
    #nearest-panel {{
      width: 360px;
      flex: 0 0 360px;
      min-width: 300px;
      display: flex;
      flex-direction: column;
      gap: 18px;
      padding: 18px;
      border-left: 1px solid rgba(148, 163, 184, 0.18);
      background: #0f172a;
      overflow-y: auto;
    }}
    .panel-kicker {{
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: #60a5fa;
      margin-bottom: 6px;
    }}
    .panel-title {{
      margin: 0;
      font-size: 28px;
      line-height: 1.1;
    }}
    .panel-caption {{
      margin: 8px 0 0;
      color: #94a3b8;
      font-size: 14px;
      line-height: 1.45;
    }}
    .neighbor-controls {{
      display: flex;
      flex-direction: column;
      gap: 10px;
      padding: 14px;
      border-radius: 14px;
      background: rgba(30, 41, 59, 0.55);
      border: 1px solid rgba(148, 163, 184, 0.12);
    }}
    .neighbor-controls-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 14px;
      color: #cbd5e1;
    }}
    .metric-badge {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-size: 12px;
      color: #94a3b8;
    }}
    .metric-badge strong {{
      color: #f8fafc;
      font-size: 12px;
      letter-spacing: 0.05em;
    }}
    .space-toggle {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }}
    .space-toggle button {{
      border: 1px solid rgba(148, 163, 184, 0.18);
      border-radius: 10px;
      background: rgba(15, 23, 42, 0.65);
      color: #cbd5e1;
      padding: 10px 12px;
      cursor: pointer;
      font-size: 13px;
      font-weight: 600;
    }}
    .space-toggle button.is-active {{
      background: rgba(37, 99, 235, 0.18);
      border-color: rgba(96, 165, 250, 0.65);
      color: #eff6ff;
    }}
    #nearest-neighbor-count {{
      width: 100%;
      accent-color: #2563eb;
    }}
    #nearest-selected {{
      display: flex;
      flex-direction: column;
      gap: 4px;
      padding: 14px;
      border-radius: 14px;
      background: rgba(30, 41, 59, 0.55);
      border: 1px solid rgba(148, 163, 184, 0.12);
    }}
    .selected-token {{
      font-size: 24px;
      font-weight: 700;
      word-break: break-word;
    }}
    .selected-source {{
      font-size: 13px;
      color: #94a3b8;
      word-break: break-word;
    }}
    .selected-empty {{
      color: #94a3b8;
      font-size: 14px;
      line-height: 1.45;
    }}
    #nearest-list {{
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}
    .neighbor-row {{
      border: 0;
      width: 100%;
      padding: 0;
      background: transparent;
      color: inherit;
      text-align: left;
      cursor: pointer;
    }}
    .neighbor-row-head {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 4px;
    }}
    .neighbor-name {{
      font-size: 16px;
      color: #f8fafc;
      word-break: break-word;
    }}
    .neighbor-distance {{
      flex: 0 0 auto;
      font-variant-numeric: tabular-nums;
      color: #cbd5e1;
    }}
    .neighbor-source {{
      margin-bottom: 6px;
      font-size: 12px;
      color: #94a3b8;
      word-break: break-word;
    }}
    .neighbor-bar {{
      height: 3px;
      width: 100%;
      border-radius: 999px;
      overflow: hidden;
      background: rgba(148, 163, 184, 0.16);
    }}
    .neighbor-bar-fill {{
      height: 100%;
      border-radius: inherit;
    }}
    .neighbor-rank {{
      color: #64748b;
      font-size: 12px;
      font-variant-numeric: tabular-nums;
      margin-bottom: 4px;
    }}
    .neighbor-bar-caption {{
      margin-top: 4px;
      font-size: 11px;
      color: #64748b;
    }}
    .nearest-empty {{
      color: #94a3b8;
      font-size: 14px;
      line-height: 1.45;
      padding: 6px 0;
    }}
    @media (max-width: 1100px) {{
      .app-shell {{
        height: auto;
        min-height: 100vh;
        overflow: auto;
      }}
      #content-shell {{
        flex-direction: column;
      }}
      #nearest-panel {{
        width: auto;
        min-width: 0;
        border-left: 0;
        border-top: 1px solid rgba(148, 163, 184, 0.18);
      }}
      #plot-container {{
        min-height: 55vh;
      }}
    }}
  </style>
</head>
<body>
  <div class=\"app-shell\">
    <div class=\"toolbar\">
      <div class=\"field\">
        <label for=\"token-filter-query\">Search tokens or labels</label>
        <input id=\"token-filter-query\" type=\"text\" placeholder=\"substring or regex\" spellcheck=\"false\">
      </div>
      <label class=\"checkbox\" for=\"token-filter-regex\">
        <input id=\"token-filter-regex\" type=\"checkbox\">
        Regex mode
      </label>
      <button id=\"token-filter-clear\" type=\"button\">Clear</button>
      <div class=\"status\">
        <strong id=\"token-filter-count\">Showing all points</strong>
        <div id=\"token-filter-error\" class=\"status-error\"></div>
      </div>
    </div>
    <div id=\"content-shell\">
      <div id=\"plot-container\">{plot_html}</div>
      <aside id=\"nearest-panel\">
        <div>
          <div class=\"panel-kicker\" id=\"nearest-space-kicker\">{default_payload["label"]}</div>
          <h2 class=\"panel-title\">Nearest Points</h2>
          <p class=\"panel-caption\" id=\"nearest-space-caption\">{default_payload["description"]}</p>
        </div>
        <div class=\"neighbor-controls\">
          <div class=\"space-toggle\">
            <button id=\"neighbor-space-original\" type=\"button\" data-space=\"original\" class=\"is-active\">Original</button>
            <button id=\"neighbor-space-projected\" type=\"button\" data-space=\"projected\">Projected</button>
          </div>
          <div class=\"neighbor-controls-header\">
            <span>Neighbors</span>
            <strong id=\"nearest-neighbor-count-value\">{neighbor_slider_value}</strong>
          </div>
          <input id=\"nearest-neighbor-count\" type=\"range\" min=\"1\" max=\"{neighbor_slider_max}\" value=\"{neighbor_slider_value}\" {neighbor_slider_disabled}>
          <div class=\"metric-badge\">Metric <strong id=\"nearest-metric-label\"></strong></div>
        </div>
        <div id=\"nearest-selected\">
          <div class=\"selected-empty\">Click a point in the plot to populate this panel.</div>
        </div>
        <div id=\"nearest-list\">
          <div class=\"nearest-empty\">Nearest points in {default_payload["label"].lower()} will appear here.</div>
        </div>
      </aside>
    </div>
  </div>
  <script>
    (() => {{
      const neighborPayload = {neighbor_payload_json};
      const graphDiv = document.getElementById('embedding-vis-plot');
      const queryInput = document.getElementById('token-filter-query');
      const regexInput = document.getElementById('token-filter-regex');
      const clearButton = document.getElementById('token-filter-clear');
      const countLabel = document.getElementById('token-filter-count');
      const errorLabel = document.getElementById('token-filter-error');
      const nearestCountInput = document.getElementById('nearest-neighbor-count');
      const nearestCountValue = document.getElementById('nearest-neighbor-count-value');
      const nearestMetricLabel = document.getElementById('nearest-metric-label');
      const nearestSpaceKicker = document.getElementById('nearest-space-kicker');
      const nearestSpaceCaption = document.getElementById('nearest-space-caption');
      const nearestSelected = document.getElementById('nearest-selected');
      const nearestList = document.getElementById('nearest-list');
      const spaceButtons = Array.from(document.querySelectorAll('.space-toggle button'));
      let selectedPointId = null;
      let activeNeighborSpace = String(neighborPayload.default_space || 'original');

      const baseData = graphDiv.data.map((trace) => {{
        const copyArray = (value) => Array.isArray(value) ? value.slice() : value;
        const marker = trace.marker || {{}};
        return {{
          type: trace.type,
          mode: trace.mode,
          x: copyArray(trace.x),
          y: copyArray(trace.y),
          z: copyArray(trace.z),
          text: copyArray(trace.text),
          hovertext: copyArray(trace.hovertext),
          customdata: copyArray(trace.customdata),
          marker: {{
            ...marker,
            size: copyArray(marker.size),
            color: copyArray(marker.color),
            line: marker.line ? {{ ...marker.line }} : marker.line,
          }},
          hovertemplate: trace.hovertemplate,
        }};
      }});
      const is3d = baseData.some((trace) => trace.type === 'scatter3d');

      const pointMetadata = new Map();
      for (const trace of baseData) {{
        for (const item of trace.customdata || []) {{
          if (!item || item.length < 3) {{
            continue;
          }}
          pointMetadata.set(String(item[0]), {{
            text: String(item[1] || ''),
            source: String(item[2] || ''),
          }});
        }}
      }}

      const totalPoints = baseData.reduce((sum, trace) => sum + (trace.text ? trace.text.length : 0), 0);

      function createLineTrace() {{
        if (is3d) {{
          return {{
            type: 'scatter3d',
            mode: 'lines',
            x: [],
            y: [],
            z: [],
            hoverinfo: 'skip',
            showlegend: false,
            line: {{ width: 2, color: 'rgba(96, 165, 250, 0.55)' }},
          }};
        }}
        return {{
          type: 'scatter',
          mode: 'lines',
          x: [],
          y: [],
          hoverinfo: 'skip',
          showlegend: false,
          line: {{ width: 2, color: 'rgba(96, 165, 250, 0.55)' }},
        }};
      }}

      function createHighlightTrace() {{
        if (is3d) {{
          return {{
            type: 'scatter3d',
            mode: 'markers',
            x: [],
            y: [],
            z: [],
            hoverinfo: 'skip',
            showlegend: false,
            marker: {{ size: [], color: [], opacity: 1, line: {{ width: 0 }} }},
          }};
        }}
        return {{
          type: 'scatter',
          mode: 'markers',
          x: [],
          y: [],
          hoverinfo: 'skip',
          showlegend: false,
          marker: {{ size: [], color: [], opacity: 1, line: {{ width: 0 }} }},
        }};
      }}

      function getSpacePayload() {{
        return neighborPayload.spaces[activeNeighborSpace] || neighborPayload.spaces.original;
      }}

      function syncSpaceControls() {{
        const payload = getSpacePayload();
        nearestSpaceKicker.textContent = String(payload.label || activeNeighborSpace);
        nearestSpaceCaption.textContent = String(payload.description || '');
        nearestMetricLabel.textContent = String(payload.metric || '').toUpperCase();

        const maxNeighbors = Number(payload.max_neighbors || 0);
        nearestCountInput.max = String(Math.max(1, maxNeighbors));
        nearestCountInput.disabled = maxNeighbors === 0;

        if (maxNeighbors === 0) {{
          nearestCountInput.value = '1';
          nearestCountValue.textContent = '0';
        }} else {{
          const nextValue = Math.min(
            Number(nearestCountInput.value || payload.default_neighbors || 1),
            maxNeighbors,
          );
          nearestCountInput.value = String(Math.max(1, nextValue));
          nearestCountValue.textContent = nearestCountInput.value;
        }}

        for (const button of spaceButtons) {{
          button.classList.toggle('is-active', button.dataset.space === activeNeighborSpace);
        }}
      }}

      function pointFromTrace(trace, index) {{
        if (!trace.customdata || !trace.customdata[index]) {{
          return null;
        }}
        return {{
          pointId: String(trace.customdata[index][0]),
          x: trace.x ? Number(trace.x[index]) : null,
          y: trace.y ? Number(trace.y[index]) : null,
          z: trace.z ? Number(trace.z[index]) : null,
        }};
      }}

      function buildVisiblePointMap(traces) {{
        const visiblePoints = new Map();
        for (const trace of traces) {{
          for (let index = 0; index < (trace.customdata || []).length; index += 1) {{
            const point = pointFromTrace(trace, index);
            if (point) {{
              visiblePoints.set(point.pointId, point);
            }}
          }}
        }}
        return visiblePoints;
      }}

      function getVisibleNeighborEntries(visiblePointMap) {{
        if (selectedPointId === null) {{
          return [];
        }}

        const payload = getSpacePayload();
        const allNeighbors = payload.neighbors[String(selectedPointId)] || [];
        const targetCount = Number(nearestCountInput.value || 0);
        const visibleNeighbors = [];

        for (const entry of allNeighbors) {{
          const pointId = String(entry[0]);
          const point = visiblePointMap.get(pointId);
          if (!point) {{
            continue;
          }}
          visibleNeighbors.push({{
            pointId,
            distance: Number(entry[1]),
            point,
          }});
          if (visibleNeighbors.length >= targetCount) {{
            break;
          }}
        }}

        return visibleNeighbors;
      }}

      function buildOverlayTraces(filteredBase) {{
        const visiblePointMap = buildVisiblePointMap(filteredBase);
        if (selectedPointId !== null && !visiblePointMap.has(String(selectedPointId))) {{
          selectedPointId = null;
        }}

        const lineTrace = createLineTrace();
        const highlightTrace = createHighlightTrace();
        if (selectedPointId === null) {{
          return {{ lineTrace, highlightTrace, visiblePointMap }};
        }}

        const selectedPoint = visiblePointMap.get(String(selectedPointId));
        if (!selectedPoint) {{
          return {{ lineTrace, highlightTrace, visiblePointMap }};
        }}

        const visibleNeighbors = getVisibleNeighborEntries(visiblePointMap);
        const selectedColor = '#f59e0b';
        const neighborColor = '#60a5fa';

        highlightTrace.x.push(selectedPoint.x);
        highlightTrace.y.push(selectedPoint.y);
        if (is3d) {{
          highlightTrace.z.push(selectedPoint.z);
        }}
        highlightTrace.marker.size.push(16);
        highlightTrace.marker.color.push(selectedColor);

        for (const entry of visibleNeighbors) {{
          const point = entry.point;
          lineTrace.x.push(selectedPoint.x, point.x, null);
          lineTrace.y.push(selectedPoint.y, point.y, null);
          if (is3d) {{
            lineTrace.z.push(selectedPoint.z, point.z, null);
          }}

          highlightTrace.x.push(point.x);
          highlightTrace.y.push(point.y);
          if (is3d) {{
            highlightTrace.z.push(point.z);
          }}
          highlightTrace.marker.size.push(11);
          highlightTrace.marker.color.push(neighborColor);
        }}

        return {{ lineTrace, highlightTrace, visiblePointMap }};
      }}

      function updateStatus(visibleCount) {{
        if (!queryInput.value) {{
          countLabel.textContent = `Showing all ${{totalPoints}} points`;
          return;
        }}
        countLabel.textContent = `Showing ${{visibleCount}} of ${{totalPoints}} points`;
      }}

      function buildMatcher() {{
        const query = queryInput.value.trim();
        if (!query) {{
          errorLabel.textContent = '';
          return null;
        }}
        if (regexInput.checked) {{
          try {{
            const regex = new RegExp(query, 'i');
            errorLabel.textContent = '';
            return (value) => regex.test(value);
          }} catch (error) {{
            errorLabel.textContent = error instanceof Error ? error.message : 'Invalid regular expression';
            return false;
          }}
        }}
        const normalizedQuery = query.toLowerCase();
        errorLabel.textContent = '';
        return (value) => value.toLowerCase().includes(normalizedQuery);
      }}

      function escapeHtml(value) {{
        return String(value).replace(/[&<>\"']/g, (character) => ({{
          '&': '&amp;',
          '<': '&lt;',
          '>': '&gt;',
          '"': '&quot;',
          "'": '&#39;',
        }})[character]);
      }}

      function formatDistance(distance) {{
        return Number.isFinite(distance) ? distance.toFixed(3) : 'n/a';
      }}

      function renderNearestPanel(visiblePointMap) {{
        syncSpaceControls();
        const payload = getSpacePayload();
        if (selectedPointId === null) {{
          nearestSelected.innerHTML = '<div class="selected-empty">Click a point in the plot to populate this panel.</div>';
          nearestList.innerHTML = `<div class="nearest-empty">Nearest points in ${{escapeHtml(payload.label || activeNeighborSpace)}} will appear here.</div>`;
          return;
        }}

        const selectedMetadata = pointMetadata.get(String(selectedPointId));
        if (!selectedMetadata || !visiblePointMap.has(String(selectedPointId))) {{
          nearestSelected.innerHTML = '<div class="selected-empty">The selected point is no longer available.</div>';
          nearestList.innerHTML = '<div class="nearest-empty">No neighbor data available for the selected point.</div>';
          return;
        }}

        nearestSelected.innerHTML = `
          <div class="selected-token">${{escapeHtml(selectedMetadata.text)}}</div>
          <div class="selected-source">${{escapeHtml(selectedMetadata.source)}}</div>
        `;

        const visibleNeighbors = getVisibleNeighborEntries(visiblePointMap);
        if (!visibleNeighbors.length) {{
          nearestList.innerHTML = '<div class="nearest-empty">No visible nearest neighbors remain for this point under the current filter.</div>';
          return;
        }}

        const maxDistance = Math.max(...visibleNeighbors.map((entry) => Number(entry.distance) || 0), 1e-6);
        nearestList.innerHTML = visibleNeighbors.map((entry, index) => {{
          const neighborId = String(entry.pointId);
          const distance = Number(entry.distance);
          const neighborMetadata = pointMetadata.get(neighborId) || {{ text: `#${{neighborId}}`, source: '' }};
          const width = Math.max(8, (1 - distance / maxDistance) * 100);
          const hue = 350 - Math.round((index / Math.max(visibleNeighbors.length - 1, 1)) * 140);
          return `
            <button class="neighbor-row" type="button" data-point-id="${{escapeHtml(neighborId)}}">
              <div class="neighbor-row-head">
                <span class="neighbor-name">${{escapeHtml(neighborMetadata.text)}}</span>
                <span class="neighbor-distance">${{formatDistance(distance)}}</span>
              </div>
              <div class="neighbor-rank">Rank #${{index + 1}}</div>
              <div class="neighbor-source">${{escapeHtml(neighborMetadata.source)}}</div>
              <div class="neighbor-bar">
                <div class="neighbor-bar-fill" style="width: ${{width.toFixed(1)}}%; background: hsl(${{hue}}deg 86% 61%);"></div>
              </div>
              <div class="neighbor-bar-caption">Bar = relative closeness within the current visible list. Number = actual distance.</div>
            </button>
          `;
        }}).join('');
      }}

      function filterTrace(trace, matcher) {{
        if (!matcher) {{
          return {{ ...trace }};
        }}

        const keep = [];
        for (let index = 0; index < trace.text.length; index += 1) {{
          const searchParts = [];
          if (trace.text[index]) searchParts.push(String(trace.text[index]));
          if (trace.hovertext[index]) searchParts.push(String(trace.hovertext[index]).replace(/<[^>]*>/g, ' '));
          if (trace.customdata[index]) {{
            for (const item of trace.customdata[index]) {{
              searchParts.push(String(item));
            }}
          }}
          if (matcher(searchParts.join(' '))) {{
            keep.push(index);
          }}
        }}

        const pick = (values) => Array.isArray(values) ? keep.map((index) => values[index]) : values;
        return {{
          ...trace,
          x: pick(trace.x),
          y: pick(trace.y),
          z: pick(trace.z),
          text: pick(trace.text),
          hovertext: pick(trace.hovertext),
          customdata: pick(trace.customdata),
          marker: {{
            ...trace.marker,
            size: pick(trace.marker.size),
            color: pick(trace.marker.color),
          }},
        }};
      }}

      function applyFilter() {{
        const matcher = buildMatcher();
        if (matcher === false) {{
          return;
        }}
        const filteredBase = baseData.map((trace) => filterTrace(trace, matcher));
        const visibleCount = filteredBase.reduce((sum, trace) => sum + (trace.text ? trace.text.length : 0), 0);
        const overlays = buildOverlayTraces(filteredBase);
        const plotData = filteredBase.concat([overlays.lineTrace, overlays.highlightTrace]);
        Plotly.react(graphDiv, plotData, graphDiv.layout, graphDiv._context);
        renderNearestPanel(overlays.visiblePointMap);
        updateStatus(visibleCount);
      }}

      queryInput.addEventListener('input', applyFilter);
      regexInput.addEventListener('change', applyFilter);
      nearestCountInput.addEventListener('input', applyFilter);
      for (const button of spaceButtons) {{
        button.addEventListener('click', () => {{
          activeNeighborSpace = String(button.dataset.space || 'original');
          applyFilter();
        }});
      }}
      nearestList.addEventListener('click', (event) => {{
        const row = event.target.closest('.neighbor-row');
        if (!row) {{
          return;
        }}
        selectedPointId = String(row.dataset.pointId);
        applyFilter();
      }});
      graphDiv.on('plotly_click', (event) => {{
        const point = event.points && event.points[0];
        if (!point || !point.customdata || point.customdata.length === 0) {{
          return;
        }}
        selectedPointId = String(point.customdata[0]);
        applyFilter();
      }});
      clearButton.addEventListener('click', () => {{
        queryInput.value = '';
        regexInput.checked = false;
        errorLabel.textContent = '';
        applyFilter();
      }});

      syncSpaceControls();
      applyFilter();
      updateStatus(totalPoints);
    }})();
  </script>
</body>
</html>
"""


def serialize_json_for_script(payload: Any) -> str:
    return (
        json.dumps(payload)
        .replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
