from __future__ import annotations

import numpy as np
import torch


def reduce_with_mds(
    vectors: np.ndarray,
    *,
    dimensions: int,
    metric: str,
    device_preference: str,
    random_seed: int,
    max_iter: int,
    learning_rate: float,
    tolerance: float,
) -> np.ndarray:
    device = resolve_mds_device(device_preference)
    try:
        reduced = metric_mds_torch(
            vectors,
            dimensions=dimensions,
            metric=metric,
            device=device,
            random_seed=random_seed,
            max_iter=max_iter,
            learning_rate=learning_rate,
            tolerance=tolerance,
        )
    except (RuntimeError, NotImplementedError) as error:
        if device_preference == "auto" and device.type != "cpu":
            print(f"Falling back to CPU for MDS because {device.type} failed: {error}")
            reduced = metric_mds_torch(
                vectors,
                dimensions=dimensions,
                metric=metric,
                device=torch.device("cpu"),
                random_seed=random_seed,
                max_iter=max_iter,
                learning_rate=learning_rate,
                tolerance=tolerance,
            )
            device = torch.device("cpu")
        else:
            raise

    print(f"MDS device: {device.type}")
    return reduced


def metric_mds_torch(
    vectors: np.ndarray,
    *,
    dimensions: int,
    metric: str,
    device: torch.device,
    random_seed: int,
    max_iter: int,
    learning_rate: float,
    tolerance: float,
) -> np.ndarray:
    points = torch.as_tensor(vectors, dtype=resolve_mds_dtype(device), device=device)
    if points.ndim != 2:
        raise ValueError(
            f"MDS expects a 2D matrix, received shape {tuple(points.shape)}."
        )
    if len(points) == 0:
        return np.zeros((0, dimensions), dtype=np.float32)
    if len(points) == 1:
        return np.zeros((1, dimensions), dtype=np.float32)

    upper_i, upper_j = torch.triu_indices(
        points.shape[0], points.shape[0], offset=1, device=device
    )
    target = compute_pairwise_dissimilarities_condensed(
        points, metric, upper_i, upper_j
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(random_seed)
    embedding = torch.randn(
        (points.shape[0], dimensions),
        dtype=points.dtype,
        device=device,
        generator=generator,
        requires_grad=True,
    )
    optimizer = torch.optim.AdamW([embedding], lr=learning_rate)

    previous_loss: float | None = None
    final_loss: float | None = None
    for step in range(max_iter):
        optimizer.zero_grad()
        projected_distances = torch.cdist(embedding, embedding, p=2.0)
        projected = projected_distances[upper_i, upper_j]
        difference = projected - target
        loss = torch.mean(difference.square())
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            embedding -= embedding.mean(dim=0, keepdim=True)

        loss_value = float(loss.detach().item())
        final_loss = loss_value
        if step % 150 == 0:
            if (
                previous_loss is not None
                and abs(previous_loss - loss_value) < tolerance
            ):
                print(
                    f"MDS converged in {step + 1} steps; "
                    f"stress={loss_value:.6f}; normalized_stress={compute_normalized_stress(target, projected):.6f}"
                )

                break
            previous_loss = loss_value
        if step % 300 == 0:
            print(
                f"MDS step {step + 1}/{max_iter}; "
                f"stress={loss_value:.6f}; normalized_stress={compute_normalized_stress(target, projected):.6f}"
            )
    else:
        if final_loss is not None:
            print(
                f"MDS stopped at max_iter={max_iter}; "
                f"stress={final_loss:.6f}; normalized_stress={compute_normalized_stress(target, projected):.6f}"
            )

    return embedding.detach().to(dtype=torch.float32).cpu().numpy()


def compute_pairwise_dissimilarities_condensed(
    points: torch.Tensor,
    metric: str,
    upper_i: torch.Tensor,
    upper_j: torch.Tensor,
) -> torch.Tensor:
    normalized_metric = metric.lower()

    if normalized_metric in {"euclidean", "l2", "sqeuclidean", "squared_euclidean"}:
        distances = torch.cdist(points, points, p=2.0)
        return distances[upper_i, upper_j]

    if normalized_metric in {"manhattan", "cityblock", "l1"}:
        distances = torch.cdist(points, points, p=1.0)
        return distances[upper_i, upper_j]

    if normalized_metric == "cosine":
        norms = torch.linalg.norm(points, dim=1, keepdim=True).clamp_min(1e-12)
        normalized_points = points / norms
        similarities = (normalized_points @ normalized_points.T).clamp(-1.0, 1.0)
        distances = 1.0 - similarities
        return distances[upper_i, upper_j]

    raise ValueError(
        "Unsupported MDS metric: "
        f"{metric}. Supported metrics are cosine, euclidean, sqeuclidean, manhattan, cityblock, and l1."
    )


def compute_normalized_stress(
    target_distances: torch.Tensor,
    projected_distances: torch.Tensor,
) -> float:
    numerator = torch.sum((projected_distances - target_distances).square())
    denominator = torch.sum(target_distances.square()).clamp_min(1e-12)
    return float(torch.sqrt(numerator / denominator).detach().item())


def resolve_mds_device(device_preference: str) -> torch.device:
    requested = device_preference.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_preference)


def resolve_mds_dtype(device: torch.device) -> torch.dtype:
    if device.type in {"cuda", "mps"}:
        return torch.float32
    return torch.float64
