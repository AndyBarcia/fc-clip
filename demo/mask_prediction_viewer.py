"""Streamlit app for inspecting FC-CLIP evaluation exports."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Callable, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple, TypeVar

import streamlit as st
from PIL import Image

IMPORT_ERRORS: List[str] = []

try:  # pragma: no cover - optional dependency during analysis
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - UI feedback path
    np = None  # type: ignore[assignment]
    IMPORT_ERRORS.append(
        "NumPy is required to visualize predictions. Install it with `pip install numpy`."
    )

try:  # pragma: no cover - optional dependency during analysis
    import pandas as pd
except ModuleNotFoundError as exc:  # pragma: no cover - UI feedback path
    pd = None  # type: ignore[assignment]
    IMPORT_ERRORS.append(
        "Pandas is required to render tables. Install it with `pip install pandas`."
    )

try:  # pragma: no cover - optional dependency during analysis
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError as exc:  # pragma: no cover - UI feedback path
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    IMPORT_ERRORS.append(
        "PyTorch is required to load FC-CLIP analysis exports. Install it with `pip install torch`."
    )

try:
    from detectron2.data import MetadataCatalog  # type: ignore
except Exception:  # pragma: no cover - optional dependency during analysis
    MetadataCatalog = None  # type: ignore


T = TypeVar("T")


ALPHA = 0.55
if np is not None:  # pragma: no branch - executed when numpy available
    COLOR_PALETTE = np.array(
        [
            [0.894, 0.102, 0.110],
            [0.215, 0.494, 0.721],
            [0.302, 0.686, 0.290],
            [0.596, 0.306, 0.639],
            [1.000, 0.498, 0.000],
            [1.000, 1.000, 0.200],
            [0.651, 0.337, 0.157],
            [0.969, 0.506, 0.749],
            [0.600, 0.600, 0.600],
            [0.100, 0.100, 0.100],
        ]
    )
else:  # pragma: no cover - handled during runtime with error message
    COLOR_PALETTE = None


def _get_cache_decorator() -> Callable[..., Callable[[Callable[..., T]], Callable[..., T]]]:
    """Return a cache decorator compatible with the installed Streamlit version."""

    if hasattr(st, "cache_data"):
        return st.cache_data

    if hasattr(st, "experimental_memo"):
        def _memo_wrapper(*args, **kwargs):  # type: ignore[override]
            return st.experimental_memo(*args, **kwargs)

        return _memo_wrapper

    def _identity_decorator(*_args, **_kwargs):
        def _identity(func: Callable[..., T]) -> Callable[..., T]:
            return func

        return _identity

    return _identity_decorator
cache_data = _get_cache_decorator()


@cache_data(show_spinner=False)
def _load_records_from_path(path: str) -> List[MutableMapping[str, torch.Tensor]]:
    records = torch.load(path, map_location="cpu")
    return _ensure_tensors(records)


@cache_data(show_spinner=False)
def _load_records_from_bytes(buffer: bytes) -> List[MutableMapping[str, torch.Tensor]]:
    stream = io.BytesIO(buffer)
    records = torch.load(stream, map_location="cpu")
    return _ensure_tensors(records)


def _ensure_tensors(
    records: Iterable[MutableMapping[str, torch.Tensor]]
) -> List[MutableMapping[str, torch.Tensor]]:
    normalized: List[MutableMapping[str, torch.Tensor]] = []
    for record in records:
        record_dict = dict(record)
        for key, value in list(record_dict.items()):
            if isinstance(value, torch.Tensor):
                record_dict[key] = value.cpu()
            elif isinstance(value, dict):
                record_dict[key] = {
                    sub_key: sub_value.cpu() if isinstance(sub_value, torch.Tensor) else sub_value
                    for sub_key, sub_value in value.items()
                }
        normalized.append(record_dict)
    return normalized


def _resolve_image_path(record: MutableMapping[str, torch.Tensor], root_override: str) -> Optional[Path]:
    file_name = record.get("file_name")
    if not file_name:
        return None

    candidate = Path(str(file_name))
    if candidate.exists():
        return candidate

    if root_override:
        override_root = Path(root_override)
        joined = override_root / candidate.name
        if joined.exists():
            return joined
        joined = override_root / candidate
        if joined.exists():
            return joined

    return candidate if candidate.exists() else None


def _prepare_predictions(
    record: MutableMapping[str, torch.Tensor],
    score_threshold: float,
    max_predictions: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits: torch.Tensor = record["pred_logits"]
    probs = logits.softmax(dim=-1)
    if probs.shape[-1] > 1:
        valid_probs = probs[:, :-1]
    else:
        valid_probs = probs

    scores, labels = valid_probs.max(dim=-1)
    keep = torch.nonzero(scores >= score_threshold, as_tuple=False).squeeze(1)
    if keep.numel() == 0:
        return torch.empty(0, dtype=torch.long), torch.empty(0), torch.empty(0, dtype=torch.long)

    ordered_scores, ordering = scores[keep].sort(descending=True)
    keep = keep[ordering[:max_predictions]]
    labels = labels[keep]
    scores = ordered_scores[:max_predictions]
    return keep, scores, labels


def _colorize_masks(
    image: Image.Image,
    masks: Sequence[np.ndarray],
    mask_threshold: float,
) -> Image.Image:
    if COLOR_PALETTE is None:
        return image

    if not masks:
        return image

    image_np = np.array(image).astype(np.float32) / 255.0
    overlay = image_np.copy()

    for idx, mask_prob in enumerate(masks):
        binary_mask = mask_prob >= mask_threshold
        if binary_mask.sum() == 0:
            continue

        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        overlay[binary_mask] = overlay[binary_mask] * (1.0 - ALPHA) + color * ALPHA

    overlay = (overlay * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def _resize_mask(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if mask.shape[-2:] == (height, width):
        return mask.to(dtype=torch.float32)
    resized = F.interpolate(
        mask.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return resized[0, 0]


def _get_class_names(dataset_name: str) -> Optional[Sequence[str]]:
    if not dataset_name or MetadataCatalog is None:
        return None

    try:
        metadata = MetadataCatalog.get(dataset_name)
    except KeyError:
        return None

    classes = getattr(metadata, "thing_classes", None)
    if isinstance(classes, Sequence):
        return classes
    return None


def _build_prediction_table(
    prediction_indices: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    class_names: Optional[Sequence[str]],
    assignments: Optional[torch.Tensor],
    gt_classes: Optional[torch.Tensor],
    gt_class_names: Optional[Sequence[str]],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for order, pred_idx in enumerate(prediction_indices.tolist()):
        class_id = int(labels[order].item()) if labels.numel() > 0 else None
        name = None
        if class_names is not None and class_id is not None and class_id < len(class_names):
            name = class_names[class_id]

        matched_gt_index: Optional[int] = None
        matched_gt_class_id: Optional[int] = None
        matched_gt_class_name: Optional[str] = None
        if assignments is not None and pred_idx < assignments.numel():
            assigned_value = int(assignments[pred_idx].item())
            if assigned_value >= 0:
                matched_gt_index = assigned_value
                if gt_classes is not None and matched_gt_index < gt_classes.numel():
                    matched_gt_class_id = int(gt_classes[matched_gt_index].item())
                    if (
                        gt_class_names is not None
                        and matched_gt_class_id is not None
                        and matched_gt_class_id < len(gt_class_names)
                    ):
                        matched_gt_class_name = gt_class_names[matched_gt_class_id]

        rows.append(
            {
                "rank": order + 1,
                "prediction_index": pred_idx,
                "class_id": class_id,
                "class_name": name,
                "score": float(scores[order].item()),
                "matched_gt_index": matched_gt_index,
                "matched_gt_class_id": matched_gt_class_id,
                "matched_gt_class_name": matched_gt_class_name,
            }
        )
    return pd.DataFrame(rows)


def _build_gt_table(gt_classes: Optional[torch.Tensor], class_names: Optional[Sequence[str]]) -> Optional[pd.DataFrame]:
    if gt_classes is None:
        return None
    rows: List[Dict[str, object]] = []
    for idx, class_id in enumerate(gt_classes.tolist()):
        name = None
        if class_names is not None and class_id < len(class_names):
            name = class_names[class_id]
        rows.append({"gt_index": idx, "class_id": class_id, "class_name": name})
    return pd.DataFrame(rows)


def _pairwise_cost_tables(
    pairwise_costs: Optional[Dict[str, torch.Tensor]]
) -> Dict[str, pd.DataFrame]:
    if not pairwise_costs:
        return {}
    tables: Dict[str, pd.DataFrame] = {}
    for name, matrix in pairwise_costs.items():
        array = matrix.detach().cpu().numpy()
        num_preds, num_targets = array.shape if array.ndim == 2 else (0, 0)
        columns = [f"gt_{i}" for i in range(num_targets)]
        index = [f"pred_{i}" for i in range(num_preds)]
        tables[name] = pd.DataFrame(array, index=index, columns=columns)
    return tables


def _match_pairs_table(
    match_indices: Optional[Dict[str, torch.Tensor]],
    gt_classes: Optional[torch.Tensor],
    class_names: Optional[Sequence[str]],
) -> Optional[pd.DataFrame]:
    if not match_indices:
        return None

    pred_tensor = match_indices.get("pred")
    gt_tensor = match_indices.get("gt")
    if not isinstance(pred_tensor, torch.Tensor) or not isinstance(gt_tensor, torch.Tensor):
        return None

    rows: List[Dict[str, object]] = []
    for pred_idx, gt_idx in zip(pred_tensor.tolist(), gt_tensor.tolist()):
        row: Dict[str, object] = {
            "prediction_index": pred_idx,
            "gt_index": gt_idx,
        }
        if gt_classes is not None and gt_idx < gt_classes.numel():
            class_id = int(gt_classes[gt_idx].item())
            row["gt_class_id"] = class_id
            if class_names is not None and class_id < len(class_names):
                row["gt_class_name"] = class_names[class_id]
        rows.append(row)

    if not rows:
        return None

    return pd.DataFrame(rows)


st.set_page_config(page_title="FC-CLIP Mask Analysis", layout="wide")

if IMPORT_ERRORS:
    st.title("FC-CLIP Mask Prediction Explorer")
    for message in IMPORT_ERRORS:
        st.error(message)
    st.stop()

st.title("FC-CLIP Mask Prediction Explorer")
st.markdown(
    "Load the `analysis_outputs.pth` artifact generated by the `MaskPredictionExporter` "
    "to inspect predictions, ground-truth targets, and Hungarian matching costs."
)

dataset_name = st.sidebar.text_input("Dataset name (optional)")
dataset_root = st.sidebar.text_input("Dataset root override", value="")
score_threshold = st.sidebar.slider("Score threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
mask_threshold = st.sidebar.slider("Mask threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
max_predictions = st.sidebar.number_input("Max predictions", min_value=1, max_value=200, value=25, step=1)

uploaded_file = st.file_uploader("Upload analysis artifact", type=["pth", "pt"])
path_input = st.text_input("or provide a path to analysis_outputs.pth", value="")

records: Optional[List[MutableMapping[str, torch.Tensor]]] = None

if uploaded_file is not None:
    try:
        records = _load_records_from_bytes(uploaded_file.getvalue())
    except Exception as exc:  # pragma: no cover - UI feedback path
        st.error(f"Failed to load uploaded artifact: {exc}")
elif path_input:
    expanded_path = str(Path(path_input).expanduser())
    if not Path(expanded_path).exists():
        st.error(f"Provided path does not exist: {expanded_path}")
    else:
        try:
            records = _load_records_from_path(expanded_path)
        except Exception as exc:  # pragma: no cover - UI feedback path
            st.error(f"Failed to load artifact from disk: {exc}")

if not records:
    st.info("Upload an analysis artifact or provide a path to begin.")
    st.stop()

num_records = len(records)
sample_index = st.sidebar.slider("Record index", min_value=0, max_value=num_records - 1, value=0, step=1)
record = records[sample_index]

st.header(f"Record {sample_index} / {num_records - 1}")

image_id = record.get("image_id")
file_name = record.get("file_name")
st.markdown(
    "**Image ID:** {image_id}  \\ **File:** {file_name}".format(
        image_id=image_id if image_id is not None else "N/A",
        file_name=file_name if file_name else "N/A",
    )
)

if "pred_logits" not in record or "pred_masks" not in record:
    st.error("Selected record does not contain prediction logits or masks.")
    st.stop()

class_names = _get_class_names(dataset_name)

prediction_indices, scores, labels = _prepare_predictions(record, score_threshold, max_predictions)
gt_classes = record.get("gt_classes")
matched_gt_assignments: Optional[torch.Tensor] = record.get("matched_gt_indices")

if prediction_indices.numel() == 0:
    st.warning("No predictions passed the current score threshold.")
else:
    prediction_table = _build_prediction_table(
        prediction_indices,
        scores,
        labels,
        class_names,
        matched_gt_assignments,
        gt_classes,
        class_names,
    )
    st.subheader("Top predictions")
    st.dataframe(prediction_table, use_container_width=True)

gt_table = _build_gt_table(gt_classes, class_names)
if gt_table is not None:
    st.subheader("Ground-truth instances")
    st.dataframe(gt_table, use_container_width=True)

match_table = _match_pairs_table(record.get("matched_indices"), gt_classes, class_names)
if match_table is not None:
    st.subheader("Hungarian assignments")
    st.dataframe(match_table, use_container_width=True)

pairwise_tables = _pairwise_cost_tables(record.get("pairwise_costs"))
if pairwise_tables:
    st.subheader("Pairwise Hungarian costs")
    for name, table in pairwise_tables.items():
        with st.expander(f"{name} cost matrix"):
            st.dataframe(table, use_container_width=True)

resolved_path = _resolve_image_path(record, dataset_root)
if resolved_path is None:
    st.warning("Image file could not be resolved for visualization. Adjust the dataset root override if necessary.")
else:
    with Image.open(resolved_path) as pil_image:
        pil_image = pil_image.convert("RGB")
        display_height = record.get("input_height") or pil_image.height
        display_width = record.get("input_width") or pil_image.width
        masks: List[np.ndarray] = []

        for pred_idx in prediction_indices.tolist():
            mask_tensor: torch.Tensor = record["pred_masks"][pred_idx]
            resized_mask = _resize_mask(mask_tensor, display_height, display_width).sigmoid().cpu().numpy()
            masks.append(resized_mask)

        colored = _colorize_masks(pil_image, masks, mask_threshold)

        st.subheader("Predicted mask overlay")
        st.image(colored, caption="Predictions overlaid on the input image", use_column_width=True)

        if gt_classes is not None and record.get("gt_masks") is not None:
            gt_masks_tensor: torch.Tensor = record["gt_masks"]
            gt_masks_np: List[np.ndarray] = []
            for mask_tensor in gt_masks_tensor:
                resized_mask = _resize_mask(mask_tensor, display_height, display_width)
                gt_masks_np.append(resized_mask.cpu().numpy())

            gt_overlay = _colorize_masks(pil_image, gt_masks_np, mask_threshold)
            st.subheader("Ground-truth mask overlay")
            st.image(gt_overlay, caption="Ground-truth masks", use_column_width=True)

st.caption(
    "Adjust the score and mask thresholds from the sidebar to control which predictions are visualized."
)
