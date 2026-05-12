from __future__ import annotations

import importlib
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


def build_visualization(config: VisualizerConfig) -> Path:
    records = load_embedding_records(config)
    vectors = np.asarray([record["vector"] for record in records], dtype=np.float32)
    projected = reduce_embeddings(vectors, config)
    dataframe = build_dataframe(records, projected)
    figure = create_figure(dataframe, config)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    html = build_interactive_html(figure)
    config.output_path.write_text(html, encoding="utf-8")
    print(f"Wrote interactive visualization to {config.output_path}")
    return config.output_path


def load_embedding_records(config: VisualizerConfig) -> list[dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModel.from_pretrained(config.model_name)
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
    for filtered_index in selected_indices:
        token_id, cleaned_text = filtered_tokens[filtered_index]
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
        f"{len(selected_indices)} (from {len(filtered_tokens)} regex-matched tokens)"
    )

    if config.extra_vectors_path is not None:
        records.extend(load_extra_vector_records(config, weights.shape[1]))

    return records


def collect_filtered_tokens(
    tokenizer: Any, total_points: int, config: VisualizerConfig
) -> list[tuple[int, str]]:
    matcher = build_token_regex_matcher(config.token_filter_regex)
    filtered_tokens: list[tuple[int, str]] = []
    for token_id in range(total_points):
        token_text = decode_token_text(tokenizer, token_id)
        cleaned_text = sanitize_token_text(token_text)
        if matcher is not None and not matcher.search(cleaned_text):
            continue
        filtered_tokens.append((token_id, cleaned_text))
    return filtered_tokens


def build_token_regex_matcher(pattern: str | None) -> re.Pattern[str] | None:
    if pattern is None:
        return None
    try:
        return re.compile(pattern)
    except re.error as error:
        raise ValueError(f"Invalid --token-regex pattern: {pattern!r}. {error}") from error


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
    config: VisualizerConfig, expected_width: int
) -> list[dict[str, Any]]:
    if config.extra_vectors_path is None:
        return []

    payload = torch.load(config.extra_vectors_path, map_location=resolve_tensor_load_device(config))
    vectors, texts = parse_extra_payload(payload, config)

    if vectors.ndim != 2:
        raise ValueError(
            f"Extra vectors must be a 2D tensor/array, received shape {tuple(vectors.shape)}."
        )
    if vectors.shape[1] != expected_width:
        raise ValueError(
            "Extra vectors must have the same embedding width as the model input embeddings: "
            f"expected {expected_width}, got {vectors.shape[1]}."
        )
    if len(texts) != vectors.shape[0]:
        raise ValueError(
            f"Extra vector labels count ({len(texts)}) does not match vector rows ({vectors.shape[0]})."
        )

    records: list[dict[str, Any]] = []
    for index, text in enumerate(texts):
        label = str(text)
        records.append(
            {
                "text": label,
                "label": label,
                "source": f"extra:{config.extra_vectors_path.name}",
                "vector": vectors[index].astype(np.float32, copy=False),
                "color": [245, 158, 11],
            }
        )
    return records


def parse_extra_payload(
    payload: Any, config: VisualizerConfig
) -> tuple[np.ndarray, list[str]]:
    if isinstance(payload, dict):
        if config.extra_vectors_key not in payload:
            raise ValueError(
                f"Extra vector file is missing '{config.extra_vectors_key}' in its top-level dictionary."
            )
        if config.extra_labels_key not in payload:
            raise ValueError(
                f"Extra vector file is missing '{config.extra_labels_key}' in its top-level dictionary."
            )
        raw_vectors = payload[config.extra_vectors_key]
        raw_labels = payload[config.extra_labels_key]
    elif isinstance(payload, torch.Tensor):
        raw_vectors = payload
        raw_labels = [f"extra_{index}" for index in range(payload.shape[0])]
    else:
        raise ValueError(
            "Unsupported extra vector payload. Expected either a tensor or a dictionary with "
            f"'{config.extra_vectors_key}' and '{config.extra_labels_key}'."
        )

    vectors = to_numpy_matrix(raw_vectors)
    labels = [str(item) for item in raw_labels]
    return vectors, labels


def to_numpy_matrix(raw_vectors: Any) -> np.ndarray:
    if isinstance(raw_vectors, torch.Tensor):
        return raw_vectors.detach().to(torch.float32).cpu().numpy()
    if isinstance(raw_vectors, np.ndarray):
        return raw_vectors.astype(np.float32, copy=False)
    return np.asarray(raw_vectors, dtype=np.float32)


def reduce_embeddings(vectors: np.ndarray, config: VisualizerConfig) -> np.ndarray:
    processed = maybe_apply_pca(vectors, config)
    if config.reduction_method == "umap":
        return reduce_with_umap(processed, config)
    if config.reduction_method == "mds":
        return reduce_with_mds(processed, config)
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


def reduce_with_mds(vectors: np.ndarray, config: VisualizerConfig) -> np.ndarray:
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
    dataframe = pd.DataFrame(
        {
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
    point_text = dataframe["text"].tolist()

    if config.dimensions == 2:
        return create_2d_figure(positions, marker_sizes, colors, hover_text, point_text)

    return create_3d_figure(
        positions, marker_sizes, colors, hover_text, point_text, config
    )


def create_2d_figure(
    positions: np.ndarray,
    marker_sizes: np.ndarray,
    colors: list[str],
    hover_text: list[str],
    point_text: list[str],
) -> go.Figure:
    scatter = go.Scatter(
        x=positions[:, 0],
        y=positions[:, 1],
        mode="markers",
        text=point_text,
        customdata=build_customdata(point_text, hover_text),
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
    config: VisualizerConfig,
) -> go.Figure:
    scatter = go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode="markers",
        text=point_text,
        customdata=build_customdata(point_text, hover_text),
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


def build_customdata(point_text: list[str], hover_text: list[str]) -> list[list[str]]:
    return [[text, hover] for text, hover in zip(point_text, hover_text)]


def build_interactive_html(figure: go.Figure) -> str:
    plot_html = figure.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={
            "displayModeBar": True,
            "scrollZoom": True,
        },
        div_id="embedding-vis-plot",
    )
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
    <div id=\"plot-container\">{plot_html}</div>
  </div>
  <script>
    (() => {{
      const graphDiv = document.getElementById('embedding-vis-plot');
      const queryInput = document.getElementById('token-filter-query');
      const regexInput = document.getElementById('token-filter-regex');
      const clearButton = document.getElementById('token-filter-clear');
      const countLabel = document.getElementById('token-filter-count');
      const errorLabel = document.getElementById('token-filter-error');

      const originalData = graphDiv.data.map((trace) => {{
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

      const totalPoints = originalData.reduce((sum, trace) => sum + (trace.text ? trace.text.length : 0), 0);

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
        const filteredData = originalData.map((trace) => filterTrace(trace, matcher));
        const visibleCount = filteredData.reduce((sum, trace) => sum + (trace.text ? trace.text.length : 0), 0);
        Plotly.react(graphDiv, filteredData, graphDiv.layout, graphDiv._context);
        updateStatus(visibleCount);
      }}

      queryInput.addEventListener('input', applyFilter);
      regexInput.addEventListener('change', applyFilter);
      clearButton.addEventListener('click', () => {{
        queryInput.value = '';
        regexInput.checked = false;
        errorLabel.textContent = '';
        applyFilter();
      }});

      updateStatus(totalPoints);
    }})();
  </script>
</body>
</html>
"""
