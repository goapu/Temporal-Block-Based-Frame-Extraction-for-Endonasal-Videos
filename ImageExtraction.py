"""
Endoscopic Frame Extraction Script
----------------------------------

Extracts classification frames from endoscopic videos using surgeon-annotated anatomical time blocks from an Excel file.

## Features

* Command-line paths and configurable extraction FPS
* Sequential OpenCV decoding to avoid keyframe-seeking label drift
* Dynamic Excel grouping by Patient + Video Name
* Canonical label normalization and skipping of non-training labels
* Optional boundary trimming, quality filtering, and face filtering
* Resume-safe extraction with per-block metadata
* Accepted/rejected frame manifests for auditability
* Multiprocessing with safe logging and crash protection
* Timestamp-driven extraction support for variable-frame-rate videos
* Incremental CSV merging to reduce memory usage

## Author

Dilip Goswami, MSc
TU Berlin, Germany
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import logging
import math
import os
import re
from collections import defaultdict
from multiprocessing import Pool, cpu_count, Manager
from pathlib import Path
from logging.handlers import QueueHandler, QueueListener
from typing import Any

import cv2
import pandas as pd


# ============================================================
# GLOBAL WORKER CACHE FOR FACE CASCADES
# ============================================================

_FACE_CASCADE_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Publication-grade endoscopic frame extraction from "
            "Excel-annotated anatomical timelines."
        )
    )

    parser.add_argument(
        "--excel_path",
        type=str,
        required=True,
        help="Path to the Excel file containing annotated video blocks."
    )

    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="Root directory containing source videos."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for frames, metadata, logs, and manifests."
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="Target extraction FPS. Examples: 2, 3, 5. Default: 5."
    )

    parser.add_argument(
        "--boundary_buffer_sec",
        type=float,
        default=0.0,
        help=(
            "Optional trim applied to both the start and end of every "
            "annotated block. Default: 0.0."
        )
    )

    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=95,
        help="JPEG quality from 1 to 100. Default: 95."
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker processes. 0 means automatic."
    )

    parser.add_argument(
        "--force_reextract",
        action="store_true",
        help="Ignore completed metadata and regenerate all matched blocks."
    )

    # --------------------------------------------------------
    # Quality filtering
    # --------------------------------------------------------

    parser.add_argument(
        "--enable_quality_filter",
        action="store_true",
        help="Enable conservative garbage-frame filtering."
    )

    parser.add_argument(
        "--min_mean_intensity",
        type=float,
        default=15.0,
        help="Reject frames darker than this mean intensity. Default: 15."
    )

    parser.add_argument(
        "--max_mean_intensity",
        type=float,
        default=240.0,
        help="Reject frames brighter than this mean intensity. Default: 240."
    )

    parser.add_argument(
        "--min_laplacian_var",
        type=float,
        default=50.0,
        help="Reject frames with Laplacian variance below this blur threshold. Default: 50."
    )

    # --------------------------------------------------------
    # Face filtering
    # --------------------------------------------------------

    parser.add_argument(
        "--enable_face_filter",
        action="store_true",
        help=(
            "Enable Haar-cascade face rejection before saving extracted frames. "
            "By default, detection runs only on Excel intervals whose raw label contains FACE."
        )
    )

    parser.add_argument(
        "--face_filter_scope",
        "--face_filter_mode",
        dest="face_filter_scope",
        type=str,
        default="marked",
        choices=["marked", "all"],
        help=(
            "Face-filter scope when --enable_face_filter is active. "
            "'marked' applies face detection only to FACE-marked Excel intervals; "
            "'all' applies it to every target frame. Default: marked."
        )
    )

    parser.add_argument(
        "--frontal_cascade_path",
        type=str,
        default="",
        help=(
            "Optional path to haarcascade_frontalface_default.xml. "
            "If omitted, OpenCV's built-in cascade path is used when available."
        )
    )

    parser.add_argument(
        "--profile_cascade_path",
        type=str,
        default="",
        help=(
            "Optional path to haarcascade_profileface.xml. "
            "If omitted, OpenCV's built-in cascade path is used when available."
        )
    )

    parser.add_argument(
        "--face_scale_factor",
        type=float,
        default=1.1,
        help="Haar cascade scaleFactor. Default: 1.1."
    )

    parser.add_argument(
        "--face_min_neighbors",
        type=int,
        default=5,
        help="Haar cascade minNeighbors. Default: 5."
    )

    parser.add_argument(
        "--face_min_size",
        type=int,
        default=100,
        help="Minimum face box size in pixels. Default: 100."
    )

    # --------------------------------------------------------
    # Timing robustness
    # --------------------------------------------------------

    parser.add_argument(
        "--timing_mode",
        type=str,
        default="presentation_time",
        choices=["presentation_time", "frame_index"],
        help=(
            "Frame matching mode. 'presentation_time' uses CAP_PROP_POS_MSEC "
            "during sequential decoding and is safer for VFR recordings. "
            "'frame_index' uses round(timestamp * source_fps). "
            "Default: presentation_time."
        )
    )

    parser.add_argument(
        "--two_field_time_format",
        type=str,
        default="mm:ss",
        choices=["mm:ss", "hh:mm", "reject"],
        help=(
            "Interpretation for ambiguous two-field text times such as 01:30. "
            "'mm:ss' means 1 minute 30 seconds, 'hh:mm' means 1 hour 30 minutes, "
            "and 'reject' skips ambiguous two-field strings. Default: mm:ss."
        )
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> bool:
    if not os.path.isfile(args.excel_path):
        print(f"ERROR: Excel file not found: {args.excel_path}")
        return False

    if not os.path.isdir(args.video_dir):
        print(f"ERROR: Video directory not found: {args.video_dir}")
        return False

    if args.fps <= 0:
        print("ERROR: --fps must be greater than 0.")
        return False

    if args.boundary_buffer_sec < 0:
        print("ERROR: --boundary_buffer_sec must be >= 0.")
        return False

    if not (1 <= args.jpeg_quality <= 100):
        print("ERROR: --jpeg_quality must be between 1 and 100.")
        return False

    if args.workers < 0:
        print("ERROR: --workers must be >= 0.")
        return False

    if args.min_mean_intensity < 0:
        print("ERROR: --min_mean_intensity must be >= 0.")
        return False

    if args.max_mean_intensity > 255:
        print("ERROR: --max_mean_intensity must be <= 255.")
        return False

    if args.min_mean_intensity >= args.max_mean_intensity:
        print("ERROR: --min_mean_intensity must be lower than --max_mean_intensity.")
        return False

    if args.min_laplacian_var < 0:
        print("ERROR: --min_laplacian_var must be >= 0.")
        return False

    if args.face_scale_factor <= 1.0:
        print("ERROR: --face_scale_factor must be greater than 1.0.")
        return False

    if args.face_min_neighbors < 0:
        print("ERROR: --face_min_neighbors must be >= 0.")
        return False

    if args.face_min_size <= 0:
        print("ERROR: --face_min_size must be > 0.")
        return False

    return True


# ============================================================
# LOGGING
# ============================================================

def configure_queue_logging(log_queue) -> None:
    """
    Route the current process logger through a multiprocessing-safe queue.
    """
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(QueueHandler(log_queue))


def start_log_listener(log_queue, log_path: str) -> QueueListener:
    """
    Start a main-process QueueListener that serializes worker logs.
    """
    formatter = logging.Formatter(
        "%(asctime)s [%(processName)s] %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    listener = QueueListener(
        log_queue,
        file_handler,
        stream_handler,
        respect_handler_level=True,
    )
    listener.start()
    return listener


def init_worker_logging(log_queue) -> None:
    configure_queue_logging(log_queue)


# ============================================================
# BASIC UTILITIES
# ============================================================

def round_float(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def normalize_video_stem(video_name: Any) -> str:
    return os.path.splitext(str(video_name).strip())[0]


def time_to_seconds(
    value: Any,
    warn: bool = True,
    two_field_time_format: str = "mm:ss",
) -> float | None:
    """
    Convert Excel-style time values into seconds.

    Supports:
    - numeric seconds
    - datetime.time
    - datetime.datetime / pandas Timestamp
    - MM:SS or HH:MM (controlled by two_field_time_format)
    - HH:MM:SS
    - HH:MM:SS.fraction

    Parameters
    ----------
    two_field_time_format:
        "mm:ss" -> 01:30 means 90 seconds
        "hh:mm" -> 01:30 means 5400 seconds
        "reject" -> ambiguous two-field strings are skipped

    Set warn=False for internal row-type scoring so expected label strings
    such as "BLOCK 1" do not flood the process log.
    """
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, datetime.datetime):
        return (
            value.hour * 3600.0
            + value.minute * 60.0
            + value.second
            + value.microsecond / 1_000_000.0
        )

    if isinstance(value, datetime.time):
        return (
            value.hour * 3600.0
            + value.minute * 60.0
            + value.second
            + value.microsecond / 1_000_000.0
        )

    value_text = str(value).strip()
    if not value_text:
        return None

    # Tolerate pandas-like timedelta text where applicable.
    if "day" in value_text.lower():
        try:
            return float(pd.to_timedelta(value_text).total_seconds())
        except Exception:
            pass

    parts = value_text.split(":")

    try:
        if len(parts) == 2:
            if two_field_time_format == "reject":
                if warn:
                    logging.warning(
                        f"⚠️ Ambiguous two-field time '{value}' rejected. "
                        "Use HH:MM:SS, MM:SS, or set --two_field_time_format explicitly."
                    )
                return None

            first = float(parts[0]) if parts[0] else 0.0
            second = float(parts[1]) if parts[1] else 0.0

            if two_field_time_format == "hh:mm":
                return first * 3600.0 + second * 60.0

            # Default and historical dataset convention.
            return first * 60.0 + second

        if len(parts) == 3:
            hours = float(parts[0]) if parts[0] else 0.0
            minutes = float(parts[1]) if parts[1] else 0.0
            seconds = float(parts[2]) if parts[2] else 0.0
            return hours * 3600.0 + minutes * 60.0 + seconds

        return float(value_text)

    except Exception as exc:
        if warn:
            logging.warning(f"⚠️ Failed to parse time value '{value}': {exc}")
        return None


def canonicalize_label(raw_label: Any) -> tuple[str | None, bool, str | None]:
    """
    Convert Excel label variants into the three training classes.

    Examples:
    - BLOCK 1, BLOCk 1, FACE + BLOCK 1, FACE + BLOC 1 -> BLOCK_1
    - BLOCK 2, BLOCK 2 + FACE -> BLOCK_2
    - BLOCK 3, BLOC 3 + FACE, BLOCK 3 LEFT + FACE -> BLOCK_3
    - MIRE + FACE or labels without a recognized cavity block -> skipped

    Returns:
    - canonical_label: str | None
    - face_marked_interval: bool
    - skip_reason: str | None
    """
    if raw_label is None:
        return None, False, "empty_label"

    try:
        if pd.isna(raw_label):
            return None, False, "empty_label"
    except Exception:
        pass

    raw_text = str(raw_label).strip()
    if not raw_text:
        return None, False, "empty_label"

    compact = re.sub(r"\s+", " ", raw_text.upper())
    face_marked_interval = "FACE" in compact

    # Harmonize known Excel variants.
    compact = re.sub(r"\bBLOC\b", "BLOCK", compact)
    compact = compact.replace("BLOCk".upper(), "BLOCK")

    if "MIRE" in compact:
        return None, face_marked_interval, "non_training_mire_label"

    match = re.search(r"\bBLOCK\s*([123])\b", compact)
    if not match:
        return None, face_marked_interval, "unrecognized_training_label"

    canonical_label = f"BLOCK_{match.group(1)}"
    return canonical_label, face_marked_interval, None


def list_jpg_files(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        return []

    return sorted(
        name for name in os.listdir(folder)
        if name.lower().endswith(".jpg")
    )


def remove_previous_block_outputs(folder: str) -> None:
    """
    Remove old frame files and per-block metadata before re-extraction.
    """
    if not os.path.isdir(folder):
        return

    removable_names = {
        "metadata.json",
        "accepted_frame_manifest.csv",
        "rejected_frame_manifest.csv",
    }

    for name in os.listdir(folder):
        path = os.path.join(folder, name)

        if name.lower().endswith(".jpg") or name in removable_names:
            try:
                os.remove(path)
            except Exception as exc:
                logging.warning(f"⚠️ Could not remove old output {path}: {exc}")


def get_video_file_signature(video_path: str) -> dict[str, int]:
    stat = os.stat(video_path)
    return {
        "source_file_size_bytes": int(stat.st_size),
        "source_file_mtime_ns": int(stat.st_mtime_ns),
    }


# ============================================================
# EXCEL PROCESSING
# ============================================================

def standardize_excel_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize columns, remove fully empty rows, identify patient/video columns.
    """
    df = df.copy()
    df.dropna(how="all", inplace=True)
    df.columns = [str(col).strip().lower() for col in df.columns]

    patient_candidates = [col for col in df.columns if "patient" in col]
    video_candidates = [
        col for col in df.columns
        if "video" in col and "name" in col
    ]

    if not patient_candidates:
        raise ValueError("No patient column found in Excel file.")

    if not video_candidates:
        raise ValueError("No video file name column found in Excel file.")

    rename_map = {
        patient_candidates[0]: "Patient",
        video_candidates[0]: "Video File Name",
    }

    df.rename(columns=rename_map, inplace=True)

    df["Patient"] = df["Patient"].ffill()
    df["Video File Name"] = df["Video File Name"].ffill()

    return df


def row_time_score(
    row: pd.Series,
    timing_columns: list[str],
    two_field_time_format: str,
) -> int:
    """
    Count how many cells in a row look like valid times.
    """
    score = 0

    for col in timing_columns:
        value = row[col]
        if pd.isna(value):
            continue

        parsed = time_to_seconds(value, warn=False, two_field_time_format=two_field_time_format)
        if parsed is not None:
            score += 1

    return score


def row_label_score(
    row: pd.Series,
    timing_columns: list[str],
    two_field_time_format: str,
) -> int:
    """
    Count how many start-column cells look like meaningful labels.
    """
    score = 0

    for col_idx in range(0, len(timing_columns) - 1, 2):
        start_col = timing_columns[col_idx]
        value = row[start_col]

        if pd.isna(value):
            continue

        label, _, skip_reason = canonicalize_label(value)
        if label is not None or skip_reason == "non_training_mire_label":
            score += 1
            continue

        parsed_time = time_to_seconds(value, warn=False, two_field_time_format=two_field_time_format)
        if parsed_time is None:
            score += 1

    return score


def identify_time_and_label_rows(
    group: pd.DataFrame,
    patient: str,
    video: str,
    timing_columns: list[str],
    two_field_time_format: str,
):
    """
    Dynamically determine which row is the timing row and which is the label row.

    Expected valid group:
    - exactly 2 rows for a patient/video pair
    - one row dominated by time-like values
    - one row dominated by label-like values
    """
    if len(group) != 2:
        logging.error(
            f"❌ Expected exactly 2 rows for {patient}/{video}, "
            f"found {len(group)}. Group skipped."
        )
        return None, None

    row_a = group.iloc[0]
    row_b = group.iloc[1]

    time_score_a = row_time_score(row_a, timing_columns, two_field_time_format)
    time_score_b = row_time_score(row_b, timing_columns, two_field_time_format)

    label_score_a = row_label_score(row_a, timing_columns, two_field_time_format)
    label_score_b = row_label_score(row_b, timing_columns, two_field_time_format)

    # Prefer a clear asymmetric assignment.
    if time_score_a > time_score_b and label_score_b >= label_score_a:
        return row_a, row_b

    if time_score_b > time_score_a and label_score_a >= label_score_b:
        return row_b, row_a

    logging.error(
        f"❌ Could not reliably identify timing vs label row for "
        f"{patient}/{video}. "
        f"Scores: rowA(time={time_score_a}, label={label_score_a}), "
        f"rowB(time={time_score_b}, label={label_score_b}). Group skipped."
    )
    return None, None


def build_timeline(
    df: pd.DataFrame,
    two_field_time_format: str,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """
    Build a validated extraction timeline using dynamic grouping by
    Patient + Video File Name.

    This avoids relying on fixed row adjacency in the raw Excel file.
    """
    timeline: dict[tuple[str, str], list[dict[str, Any]]] = {}

    timing_columns = [
        col for col in df.columns
        if col not in {"Patient", "Video File Name"}
    ]

    if len(timing_columns) % 2 != 0:
        logging.warning(
            "⚠️ Timing columns are not an even start/end sequence. "
            "The final unmatched column will be ignored."
        )

    grouped = df.groupby(
        ["Patient", "Video File Name"],
        sort=False,
        dropna=False,
    )

    for (patient_raw, video_raw), group in grouped:
        patient = str(patient_raw).strip().upper()
        video = normalize_video_stem(video_raw)

        if not patient or patient == "NAN":
            logging.error("❌ Invalid patient identifier. Group skipped.")
            continue

        if not video or video.lower() == "nan":
            logging.error(f"❌ Invalid video identifier for patient {patient}. Group skipped.")
            continue

        if "VIDEO FILES ALREADY SEPARATED IN BLOCKS" in video.upper():
            logging.info(f"ℹ️ Skipping non-timeline note row for {patient}/{video}.")
            continue

        time_row, label_row = identify_time_and_label_rows(
            group=group,
            patient=patient,
            video=video,
            timing_columns=timing_columns,
            two_field_time_format=two_field_time_format,
        )

        if time_row is None or label_row is None:
            continue

        blocks: list[dict[str, Any]] = []

        for col_idx in range(0, len(timing_columns) - 1, 2):
            start_col = timing_columns[col_idx]
            end_col = timing_columns[col_idx + 1]

            raw_label = label_row[start_col]
            raw_start = time_row[start_col]
            raw_end = time_row[end_col]

            if pd.isna(raw_label) or pd.isna(raw_start) or pd.isna(raw_end):
                continue

            label, face_marked_interval, skip_reason = canonicalize_label(raw_label)

            if label is None:
                logging.info(
                    f"ℹ️ Skipping label '{raw_label}' for {patient}/{video}: {skip_reason}."
                )
                continue

            start_sec = time_to_seconds(
                raw_start,
                two_field_time_format=two_field_time_format,
            )
            end_sec = time_to_seconds(
                raw_end,
                two_field_time_format=two_field_time_format,
            )

            if start_sec is None or end_sec is None:
                logging.warning(
                    f"⚠️ Time parsing failed for {patient}/{video} [{label}]. "
                    f"Raw values: {raw_start} -> {raw_end}. Block skipped."
                )
                continue

            if start_sec < 0 or end_sec < 0:
                logging.warning(
                    f"⚠️ Negative timestamps for {patient}/{video} [{label}]: "
                    f"{start_sec} -> {end_sec}. Block skipped."
                )
                continue

            if start_sec >= end_sec:
                logging.warning(
                    f"⚠️ Invalid block duration for {patient}/{video} [{label}]: "
                    f"{start_sec:.3f}s -> {end_sec:.3f}s. Block skipped."
                )
                continue

            blocks.append(
                {
                    "class_label": label,
                    "raw_label": str(raw_label),
                    "face_marked_interval": bool(face_marked_interval),
                    "raw_start": str(raw_start),
                    "raw_end": str(raw_end),
                    "start_sec": float(start_sec),
                    "end_sec": float(end_sec),
                }
            )

        blocks.sort(key=lambda block: block["start_sec"])

        valid_blocks: list[dict[str, Any]] = []

        for block in blocks:
            if valid_blocks:
                previous = valid_blocks[-1]

                if block["start_sec"] < previous["end_sec"]:
                    logging.error(
                        f"❌ Overlap detected in {video}: "
                        f"{previous['class_label']} ends at "
                        f"{previous['end_sec']:.3f}s, but "
                        f"{block['class_label']} starts at "
                        f"{block['start_sec']:.3f}s. "
                        f"Skipping {block['class_label']}."
                    )
                    continue

                gap = block["start_sec"] - previous["end_sec"]

                if gap > 0:
                    logging.info(
                        f"🛑 Preserved no-extraction gap of {gap:.3f}s in {video}: "
                        f"{previous['class_label']} -> {block['class_label']}."
                    )

            valid_blocks.append(block)

        if valid_blocks:
            timeline[(patient, video)] = valid_blocks
        else:
            logging.info(f"ℹ️ No valid training blocks retained for {patient}/{video}.")

    return timeline


# ============================================================
# VIDEO DISCOVERY
# ============================================================

def discover_video_tasks(
    timeline: dict[tuple[str, str], list[dict[str, Any]]],
    video_dir: str,
    output_dir: str,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    matched_keys: set[tuple[str, str]] = set()
    supported_exts = {".mp4", ".mov", ".m4v"}

    for root, _, files in os.walk(video_dir):
        for filename in files:
            if filename.startswith("._"):
                continue

            suffix = os.path.splitext(filename)[1].lower()
            if suffix not in supported_exts:
                continue

            video_stem = os.path.splitext(filename)[0]
            patient = video_stem.split("_")[0].upper()
            key = (patient, video_stem)

            if key not in timeline:
                continue

            tasks.append(
                {
                    "patient": patient,
                    "video": video_stem,
                    "video_path": os.path.join(root, filename),
                    "output_dir": output_dir,
                    "blocks": timeline[key],
                    "config": config,
                }
            )

            matched_keys.add(key)
            logging.info(f"✅ Matched video: {key}")

    unmatched = sorted(set(timeline.keys()) - matched_keys)

    if unmatched:
        unmatched_path = os.path.join(
            output_dir,
            "timeline_entries_without_matching_video.txt",
        )

        with open(unmatched_path, "w", encoding="utf-8") as handle:
            for patient, video in unmatched:
                handle.write(f"{patient}\t{video}\n")

        logging.warning(
            f"⚠️ {len(unmatched)} timeline entries had no matched video. "
            f"Saved to {unmatched_path}"
        )

    return tasks


# ============================================================
# FPS AND EXTRACTION PLAN
# ============================================================

def generate_target_timestamps(
    start_sec: float,
    end_sec: float,
    target_fps: float,
) -> list[float]:
    """
    Generate timestamps in [start_sec, end_sec) at runtime-selected FPS.
    """
    if end_sec <= start_sec:
        return []

    interval = 1.0 / target_fps
    duration = end_sec - start_sec
    count = int(math.floor(duration * target_fps + 1e-9))

    timestamps = [
        start_sec + i * interval
        for i in range(count)
    ]

    return [
        timestamp for timestamp in timestamps
        if timestamp < end_sec - 1e-9
    ]


def timestamps_to_unique_frame_indices(
    timestamps: list[float],
    source_fps: float,
    total_frames: int,
) -> list[tuple[float, int]]:
    """
    Convert timestamps to unique source-video frame indices.
    """
    pairs: list[tuple[float, int]] = []
    seen: set[int] = set()

    for timestamp in timestamps:
        frame_idx = int(round(timestamp * source_fps))

        if frame_idx < 0:
            continue

        if total_frames > 0:
            frame_idx = min(frame_idx, total_frames - 1)

        if frame_idx in seen:
            continue

        seen.add(frame_idx)
        pairs.append((timestamp, frame_idx))

    return pairs


def capture_timestamp_seconds(
    cap,
    current_frame_idx: int,
    source_fps: float,
    last_timestamp: float | None = None,
) -> tuple[float, str]:
    """
    Return the best available timestamp for the most recently grabbed frame.

    Priority:
    1. CAP_PROP_POS_MSEC when it is finite, non-negative, and monotonic.
    2. frame_index / FPS fallback when POS_MSEC is unavailable or non-monotonic.
    """
    pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)

    if isinstance(pos_msec, (int, float)) and math.isfinite(pos_msec) and pos_msec >= 0:
        timestamp = float(pos_msec) / 1000.0
        if last_timestamp is None or timestamp + 1e-9 >= last_timestamp:
            return timestamp, "cap_pos_msec"

    return current_frame_idx / source_fps, "frame_index_fallback"


# ============================================================
# FACE FILTERING
# ============================================================

def resolve_cascade_path(user_path: str, default_filename: str) -> str:
    """
    Resolve Haar cascade XML paths robustly.

    Priority:
    1. Explicit CLI path.
    2. XML file placed next to this script.
    3. OpenCV's bundled cv2.data.haarcascades directory.

    The caller still validates whether the returned XML actually loads.
    """
    if user_path:
        return user_path

    script_dir_candidate = Path(__file__).resolve().parent / default_filename
    if script_dir_candidate.is_file():
        return str(script_dir_candidate)

    builtin_dir = getattr(cv2.data, "haarcascades", "")
    return os.path.join(builtin_dir, default_filename)


def load_face_cascades(config: dict[str, Any]):
    """
    Load Haar cascades once per worker and cache them.
    """
    global _FACE_CASCADE_CACHE

    if not config["enable_face_filter"]:
        return None, None

    frontal_path = resolve_cascade_path(
        config["frontal_cascade_path"],
        "haarcascade_frontalface_default.xml",
    )

    profile_path = resolve_cascade_path(
        config["profile_cascade_path"],
        "haarcascade_profileface.xml",
    )

    cache_key = (frontal_path, profile_path)

    if cache_key in _FACE_CASCADE_CACHE:
        return _FACE_CASCADE_CACHE[cache_key]

    frontal_cascade = cv2.CascadeClassifier(frontal_path)
    profile_cascade = cv2.CascadeClassifier(profile_path)

    if frontal_cascade.empty():
        logging.warning(
            f"⚠️ Frontal face cascade could not be loaded: {frontal_path}. "
            "Place haarcascade_frontalface_default.xml next to the script "
            "or pass --frontal_cascade_path explicitly."
        )
        frontal_cascade = None

    if profile_cascade.empty():
        logging.warning(
            f"⚠️ Profile face cascade could not be loaded: {profile_path}. "
            "Place haarcascade_profileface.xml next to the script "
            "or pass --profile_cascade_path explicitly."
        )
        profile_cascade = None

    _FACE_CASCADE_CACHE[cache_key] = (frontal_cascade, profile_cascade)
    return frontal_cascade, profile_cascade


def detect_face(
    frame,
    frontal_cascade,
    profile_cascade,
    scale_factor: float,
    min_neighbors: int,
    min_size: int,
) -> tuple[bool, int, int]:
    """
    Return:
    - face_detected
    - frontal_count
    - profile_count
    """
    if frontal_cascade is None and profile_cascade is None:
        return False, 0, 0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frontal_count = 0
    profile_count = 0

    if frontal_cascade is not None:
        frontal_faces = frontal_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(min_size, min_size),
        )
        frontal_count = len(frontal_faces)

    if profile_cascade is not None:
        profile_faces = profile_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(min_size, min_size),
        )
        profile_count = len(profile_faces)

    detected = frontal_count > 0 or profile_count > 0
    return detected, frontal_count, profile_count


# ============================================================
# QUALITY FILTERING
# ============================================================

def evaluate_frame_quality(
    frame,
    config: dict[str, Any],
) -> tuple[bool, list[str], dict[str, float]]:
    """
    Conservative image-quality rejection.

    Returns:
    - accepted
    - rejection_reasons
    - quality_metrics
    """
    if not config["enable_quality_filter"]:
        return True, [], {}

    reasons: list[str] = []

    mean_intensity = float(frame.mean())

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if mean_intensity < config["min_mean_intensity"]:
        reasons.append("intensity_too_low")

    if mean_intensity > config["max_mean_intensity"]:
        reasons.append("intensity_too_high")

    if laplacian_var < config["min_laplacian_var"]:
        reasons.append("severe_blur")

    accepted = len(reasons) == 0

    metrics = {
        "Mean_Intensity": round_float(mean_intensity),
        "Laplacian_Variance": round_float(laplacian_var),
    }

    return accepted, reasons, metrics


# ============================================================
# METADATA / RESUME LOGIC
# ============================================================

def metadata_matches(existing: dict[str, Any], expected: dict[str, Any]) -> bool:
    exact_keys = [
        "patient",
        "video",
        "block_name",
        "class_label",
        "raw_label",
        "face_marked_interval",
        "occurrence_index",
        "planned_frame_count",
        "source_file_size_bytes",
        "source_file_mtime_ns",
        "jpeg_quality",
        "enable_quality_filter",
        "enable_face_filter",
        "face_filter_scope",
        "timing_mode",
        "completed",
    ]

    for key in exact_keys:
        if existing.get(key) != expected.get(key):
            return False

    numeric_keys = [
        "original_start_sec",
        "original_end_sec",
        "buffered_start_sec",
        "buffered_end_sec",
        "target_fps",
        "boundary_buffer_sec",
        "source_video_fps",
        "min_mean_intensity",
        "max_mean_intensity",
        "min_laplacian_var",
        "face_scale_factor",
        "face_min_neighbors",
        "face_min_size",
    ]

    for key in numeric_keys:
        if key not in existing or key not in expected:
            return False

        if abs(float(existing[key]) - float(expected[key])) > 1e-5:
            return False

    return existing.get("completed") is True


# ============================================================
# SINGLE-VIDEO PROCESSING
# ============================================================

def process_video(task: dict[str, Any]):
    patient = task["patient"]
    video = task["video"]
    video_path = task["video_path"]
    output_dir = task["output_dir"]
    blocks = task["blocks"]
    config = task["config"]

    target_fps = float(config["target_fps"])
    boundary_buffer_sec = float(config["boundary_buffer_sec"])
    jpeg_quality = int(config["jpeg_quality"])
    force_reextract = bool(config["force_reextract"])
    timing_mode = str(config["timing_mode"])

    block_summaries: list[dict[str, Any]] = []
    accepted_manifest_paths: list[str] = []
    rejected_manifest_paths: list[str] = []
    unreadable_videos: list[str] = []

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logging.error(f"❌ Cannot open video: {video_path}")
        unreadable_videos.append(video_path)
        return (
            block_summaries,
            accepted_manifest_paths,
            rejected_manifest_paths,
            unreadable_videos,
        )

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if not source_fps or source_fps <= 0:
        logging.error(f"❌ Invalid source FPS for video: {video_path}")
        cap.release()
        unreadable_videos.append(video_path)
        return (
            block_summaries,
            accepted_manifest_paths,
            rejected_manifest_paths,
            unreadable_videos,
        )

    has_reported_frame_count = total_frames > 0
    duration_sec = total_frames / source_fps if has_reported_frame_count else None
    duration_text = f"{duration_sec:.3f}s" if duration_sec is not None else "unknown"

    logging.info(
        f"🎞️ Processing {video}: "
        f"{total_frames if has_reported_frame_count else 'unknown'} frames, "
        f"source FPS={source_fps:.3f}, duration={duration_text}, "
        f"target FPS={target_fps}, timing_mode={timing_mode}."
    )

    if target_fps > source_fps:
        logging.warning(
            f"⚠️ Target FPS ({target_fps}) exceeds source FPS ({source_fps:.3f}) "
            f"for {video}. Duplicate frame requests may collapse."
        )

    if duration_sec is None:
        logging.warning(
            f"⚠️ OpenCV reported frame count=0 for {video}. "
            "Block duration checks will not be used; sequential decoding will "
            "stop only when cap.grab() reaches end-of-stream."
        )

    video_signature = get_video_file_signature(video_path)
    frontal_cascade, profile_cascade = load_face_cascades(config)

    label_counter: defaultdict[str, int] = defaultdict(int)
    pending_frame_targets: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    pending_time_targets: list[dict[str, Any]] = []
    pending_blocks: list[dict[str, Any]] = []

    # --------------------------------------------------------
    # Build extraction plan
    # --------------------------------------------------------

    for block in blocks:
        class_label = block["class_label"]
        raw_label = block.get("raw_label", class_label)
        face_marked_interval = bool(block.get("face_marked_interval", False))
        original_start = float(block["start_sec"])
        original_end = float(block["end_sec"])

        label_counter[class_label] += 1
        occurrence_index = label_counter[class_label]

        block_name = (
            class_label
            if occurrence_index == 1
            else f"{class_label}_{occurrence_index}"
        )

        # Only perform duration-based clipping when OpenCV reports a usable duration.
        if duration_sec is not None and original_start >= duration_sec:
            logging.warning(
                f"⏭️ Skipping {video}/{block_name}: "
                f"start {original_start:.3f}s exceeds reported video duration "
                f"{duration_sec:.3f}s."
            )
            continue

        clipped_end = min(original_end, duration_sec) if duration_sec is not None else original_end

        if duration_sec is not None and clipped_end < original_end:
            logging.warning(
                f"⚠️ Trimmed {video}/{block_name} end from "
                f"{original_end:.3f}s to {clipped_end:.3f}s."
            )

        buffered_start = original_start + boundary_buffer_sec
        buffered_end = clipped_end - boundary_buffer_sec

        if buffered_start >= buffered_end:
            logging.warning(
                f"⚠️ Skipping {video}/{block_name}: empty after "
                f"boundary buffer of {boundary_buffer_sec:.3f}s."
            )
            continue

        timestamps = generate_target_timestamps(
            start_sec=buffered_start,
            end_sec=buffered_end,
            target_fps=target_fps,
        )

        if timing_mode == "frame_index":
            extraction_targets = timestamps_to_unique_frame_indices(
                timestamps=timestamps,
                source_fps=source_fps,
                total_frames=total_frames,
            )
        else:
            extraction_targets = [(timestamp, None) for timestamp in timestamps]

        if not extraction_targets:
            logging.warning(
                f"⚠️ Skipping {video}/{block_name}: no extractable target frames."
            )
            continue

        planned_frame_count = len(extraction_targets)

        block_dir = os.path.join(output_dir, patient, video, block_name)
        os.makedirs(block_dir, exist_ok=True)

        metadata_path = os.path.join(block_dir, "metadata.json")
        accepted_manifest_path = os.path.join(
            block_dir,
            "accepted_frame_manifest.csv",
        )
        rejected_manifest_path = os.path.join(
            block_dir,
            "rejected_frame_manifest.csv",
        )

        expected_metadata = {
            "patient": patient,
            "video": video,
            "block_name": block_name,
            "class_label": class_label,
            "raw_label": str(raw_label),
            "face_marked_interval": bool(face_marked_interval),
            "occurrence_index": int(occurrence_index),
            "raw_start": block["raw_start"],
            "raw_end": block["raw_end"],
            "original_start_sec": round_float(original_start),
            "original_end_sec": round_float(original_end),
            "buffered_start_sec": round_float(buffered_start),
            "buffered_end_sec": round_float(buffered_end),
            "target_fps": round_float(target_fps),
            "boundary_buffer_sec": round_float(boundary_buffer_sec),
            "source_video_fps": round_float(source_fps),
            "source_total_frames": int(total_frames),
            "source_duration_sec": round_float(duration_sec) if duration_sec is not None else None,
            "planned_frame_count": int(planned_frame_count),
            "jpeg_quality": int(jpeg_quality),
            "enable_quality_filter": bool(config["enable_quality_filter"]),
            "min_mean_intensity": round_float(config["min_mean_intensity"]),
            "max_mean_intensity": round_float(config["max_mean_intensity"]),
            "min_laplacian_var": round_float(config["min_laplacian_var"]),
            "enable_face_filter": bool(config["enable_face_filter"]),
            "face_filter_scope": str(config["face_filter_scope"]),
            "face_scale_factor": round_float(config["face_scale_factor"]),
            "face_min_neighbors": int(config["face_min_neighbors"]),
            "face_min_size": int(config["face_min_size"]),
            "timing_mode": timing_mode,
            "completed": True,
            **video_signature,
        }

        # ----------------------------------------------------
        # Resume check
        # ----------------------------------------------------

        existing_jpgs = list_jpg_files(block_dir)

        if (
            not force_reextract
            and os.path.exists(metadata_path)
            and os.path.exists(accepted_manifest_path)
            and os.path.exists(rejected_manifest_path)
        ):
            try:
                with open(metadata_path, "r", encoding="utf-8") as handle:
                    existing_metadata = json.load(handle)

                accepted_count_existing = int(
                    existing_metadata.get("accepted_frame_count", -1)
                )

                if (
                    metadata_matches(existing_metadata, expected_metadata)
                    and len(existing_jpgs) == accepted_count_existing
                ):
                    logging.info(f"⏭️ Resume skip: {video}/{block_name} already complete.")

                    accepted_manifest_paths.append(accepted_manifest_path)
                    rejected_manifest_paths.append(rejected_manifest_path)

                    block_summaries.append(
                        {
                            "Patient": patient,
                            "Video": video,
                            "Block_Name": block_name,
                            "Class_Label": class_label,
                            "Raw_Label": str(raw_label),
                            "Face_Marked_Interval": bool(face_marked_interval),
                            "Occurrence_Index": occurrence_index,
                            "Original_Start_Sec": round_float(original_start),
                            "Original_End_Sec": round_float(original_end),
                            "Buffered_Start_Sec": round_float(buffered_start),
                            "Buffered_End_Sec": round_float(buffered_end),
                            "Target_FPS": target_fps,
                            "Boundary_Buffer_Sec": boundary_buffer_sec,
                            "Timing_Mode": timing_mode,
                            "Planned_Frames": planned_frame_count,
                            "Accepted_Frames": existing_metadata.get("accepted_frame_count", 0),
                            "Rejected_Frames": existing_metadata.get("rejected_frame_count", 0),
                            "Status": "resumed_existing",
                        }
                    )
                    continue

            except Exception as exc:
                logging.warning(
                    f"⚠️ Resume validation failed for "
                    f"{video}/{block_name}: {exc}. Re-extracting."
                )

        remove_previous_block_outputs(block_dir)

        block_plan = {
            "patient": patient,
            "video": video,
            "block_name": block_name,
            "class_label": class_label,
            "raw_label": str(raw_label),
            "face_marked_interval": bool(face_marked_interval),
            "occurrence_index": occurrence_index,
            "original_start": original_start,
            "original_end": original_end,
            "buffered_start": buffered_start,
            "buffered_end": buffered_end,
            "block_dir": block_dir,
            "metadata_path": metadata_path,
            "accepted_manifest_path": accepted_manifest_path,
            "rejected_manifest_path": rejected_manifest_path,
            "expected_metadata": expected_metadata,
            "extraction_targets": extraction_targets,
            "planned_frame_count": planned_frame_count,
            "processed_frame_count": 0,
            "accepted_frame_count": 0,
            "rejected_frame_count": 0,
            "accepted_rows": [],
            "rejected_rows": [],
        }

        pending_blocks.append(block_plan)

        if timing_mode == "frame_index":
            for requested_timestamp, source_frame_idx in extraction_targets:
                assert source_frame_idx is not None
                pending_frame_targets[int(source_frame_idx)].append(
                    {
                        "block_plan": block_plan,
                        "requested_timestamp": requested_timestamp,
                    }
                )
        else:
            for requested_timestamp, _ in extraction_targets:
                pending_time_targets.append(
                    {
                        "block_plan": block_plan,
                        "requested_timestamp": float(requested_timestamp),
                    }
                )

    # --------------------------------------------------------
    # Everything resumed
    # --------------------------------------------------------

    if not pending_blocks:
        cap.release()
        return (
            block_summaries,
            accepted_manifest_paths,
            rejected_manifest_paths,
            unreadable_videos,
        )

    # --------------------------------------------------------
    # Target processing helper
    # --------------------------------------------------------

    def process_target_frame(
        plan: dict[str, Any],
        requested_timestamp: float,
        source_frame_idx: int,
        actual_timestamp: float,
        timestamp_source: str,
        frame,
    ) -> None:
        plan["processed_frame_count"] += 1

        quality_ok, quality_reasons, quality_metrics = evaluate_frame_quality(
            frame,
            config,
        )
        rejection_reasons = list(quality_reasons)

        face_detected = False
        frontal_count = 0
        profile_count = 0

        face_filter_applied = bool(
            config["enable_face_filter"]
            and (
                config["face_filter_scope"] == "all"
                or plan["face_marked_interval"]
            )
        )

        if face_filter_applied:
            face_detected, frontal_count, profile_count = detect_face(
                frame=frame,
                frontal_cascade=frontal_cascade,
                profile_cascade=profile_cascade,
                scale_factor=config["face_scale_factor"],
                min_neighbors=config["face_min_neighbors"],
                min_size=config["face_min_size"],
            )

            if face_detected:
                rejection_reasons.append("face_detected")

        common_row = {
            "Patient": patient,
            "Video": video,
            "Block_Name": plan["block_name"],
            "Class_Label": plan["class_label"],
            "Raw_Label": plan["raw_label"],
            "Face_Marked_Interval": bool(plan["face_marked_interval"]),
            "Face_Filter_Applied": bool(face_filter_applied),
            "Occurrence_Index": plan["occurrence_index"],
            "Requested_Timestamp_Sec": round_float(requested_timestamp),
            "Actual_Frame_Timestamp_Sec": round_float(actual_timestamp),
            "Timestamp_Delta_Sec": round_float(actual_timestamp - requested_timestamp),
            "Timestamp_Source": timestamp_source,
            "Timing_Mode": timing_mode,
            "Source_Frame_Index": int(source_frame_idx),
            "Face_Detected": bool(face_detected),
            "Frontal_Face_Count": int(frontal_count),
            "Profile_Face_Count": int(profile_count),
            "Mean_Intensity": quality_metrics.get("Mean_Intensity", ""),
            "Laplacian_Variance": quality_metrics.get("Laplacian_Variance", ""),
            "Target_FPS": target_fps,
            "Boundary_Buffer_Sec": boundary_buffer_sec,
        }

        if rejection_reasons:
            plan["rejected_frame_count"] += 1
            rejected_row = {
                **common_row,
                "Rejection_Reasons": "|".join(rejection_reasons),
            }
            plan["rejected_rows"].append(rejected_row)

            logging.info(
                f"🚫 Rejected {video}/{plan['block_name']} "
                f"frame {source_frame_idx}: {'|'.join(rejection_reasons)}"
            )
            return

        plan["accepted_frame_count"] += 1
        accepted_number = plan["accepted_frame_count"]

        filename = f"{video}_{plan['block_name']}_{accepted_number:04d}.jpg"
        output_path = os.path.join(plan["block_dir"], filename)

        write_ok = cv2.imwrite(
            output_path,
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
        )

        if not write_ok:
            logging.warning(f"⚠️ Failed to write image: {output_path}")
            plan["accepted_frame_count"] -= 1
            plan["rejected_frame_count"] += 1
            rejected_row = {
                **common_row,
                "Rejection_Reasons": "write_failure",
            }
            plan["rejected_rows"].append(rejected_row)
            return

        accepted_row = {
            **common_row,
            "Filename": filename,
            "File_Path": output_path,
        }
        plan["accepted_rows"].append(accepted_row)

    # --------------------------------------------------------
    # Sequential decoding with grab/retrieve
    # --------------------------------------------------------
    # - cap.grab() for every source frame
    # - cap.retrieve() only for frames that satisfy extraction targets
    # - no cap.set(...), so no random-seek keyframe drift
    # --------------------------------------------------------

    current_frame_idx = 0
    decode_loop_failed = False
    last_capture_timestamp: float | None = None

    try:
        if timing_mode == "frame_index":
            target_indices = sorted(pending_frame_targets.keys())
            last_target_idx = target_indices[-1]

            logging.info(
                f"🧩 Frame-index extraction for {video}: "
                f"walking from source frame 0 to {last_target_idx}."
            )

            while current_frame_idx <= last_target_idx:
                ret = cap.grab()

                if not ret:
                    logging.warning(
                        f"⚠️ Grab failure/end-of-stream in {video} near source frame "
                        f"{current_frame_idx}."
                    )
                    break

                actual_timestamp, timestamp_source = capture_timestamp_seconds(
                    cap=cap,
                    current_frame_idx=current_frame_idx,
                    source_fps=source_fps,
                    last_timestamp=last_capture_timestamp,
                )
                last_capture_timestamp = actual_timestamp

                if current_frame_idx in pending_frame_targets:
                    retrieve_ok, frame = cap.retrieve()

                    if not retrieve_ok or frame is None:
                        logging.warning(
                            f"⚠️ Retrieve failure in {video} at target frame "
                            f"{current_frame_idx}."
                        )
                        current_frame_idx += 1
                        continue

                    for target in pending_frame_targets[current_frame_idx]:
                        process_target_frame(
                            plan=target["block_plan"],
                            requested_timestamp=float(target["requested_timestamp"]),
                            source_frame_idx=current_frame_idx,
                            actual_timestamp=actual_timestamp,
                            timestamp_source=timestamp_source,
                            frame=frame,
                        )

                current_frame_idx += 1

        else:
            pending_time_targets.sort(key=lambda item: item["requested_timestamp"])
            next_target_idx = 0
            total_time_targets = len(pending_time_targets)

            logging.info(
                f"🧩 Presentation-time extraction for {video}: "
                f"matching {total_time_targets} requested timestamps sequentially."
            )

            while next_target_idx < total_time_targets:
                ret = cap.grab()

                if not ret:
                    logging.warning(
                        f"⚠️ Grab failure/end-of-stream in {video} before all "
                        "timestamp targets were reached."
                    )
                    break

                actual_timestamp, timestamp_source = capture_timestamp_seconds(
                    cap=cap,
                    current_frame_idx=current_frame_idx,
                    source_fps=source_fps,
                    last_timestamp=last_capture_timestamp,
                )
                last_capture_timestamp = actual_timestamp

                due_targets: list[dict[str, Any]] = []
                while (
                    next_target_idx < total_time_targets
                    and actual_timestamp + 1e-9
                    >= float(pending_time_targets[next_target_idx]["requested_timestamp"])
                ):
                    due_targets.append(pending_time_targets[next_target_idx])
                    next_target_idx += 1

                if due_targets:
                    retrieve_ok, frame = cap.retrieve()

                    if not retrieve_ok or frame is None:
                        logging.warning(
                            f"⚠️ Retrieve failure in {video} near source frame "
                            f"{current_frame_idx} while processing timestamp targets."
                        )
                        current_frame_idx += 1
                        continue

                    for target in due_targets:
                        process_target_frame(
                            plan=target["block_plan"],
                            requested_timestamp=float(target["requested_timestamp"]),
                            source_frame_idx=current_frame_idx,
                            actual_timestamp=actual_timestamp,
                            timestamp_source=timestamp_source,
                            frame=frame,
                        )

                current_frame_idx += 1

    except cv2.error as exc:
        decode_loop_failed = True
        unreadable_videos.append(video_path)
        logging.exception(
            f"❌ OpenCV error while decoding/filtering {video}. "
            f"Partial outputs will be preserved and marked incomplete: {exc}"
        )
    except Exception as exc:
        decode_loop_failed = True
        unreadable_videos.append(video_path)
        logging.exception(
            f"❌ Unexpected error while decoding/filtering {video}. "
            f"Partial outputs will be preserved and marked incomplete: {exc}"
        )
    finally:
        cap.release()

    # --------------------------------------------------------
    # Save per-block outputs
    # --------------------------------------------------------

    for plan in pending_blocks:
        planned_count = plan["planned_frame_count"]
        processed_count = plan["processed_frame_count"]
        accepted_count = plan["accepted_frame_count"]
        rejected_count = plan["rejected_frame_count"]

        completed = (processed_count == planned_count) and not decode_loop_failed

        pd.DataFrame(plan["accepted_rows"]).to_csv(
            plan["accepted_manifest_path"],
            index=False,
        )

        pd.DataFrame(plan["rejected_rows"]).to_csv(
            plan["rejected_manifest_path"],
            index=False,
        )

        accepted_manifest_paths.append(plan["accepted_manifest_path"])
        rejected_manifest_paths.append(plan["rejected_manifest_path"])

        final_metadata = dict(plan["expected_metadata"])
        final_metadata["processed_frame_count"] = int(processed_count)
        final_metadata["accepted_frame_count"] = int(accepted_count)
        final_metadata["rejected_frame_count"] = int(rejected_count)
        final_metadata["completed"] = bool(completed)

        with open(plan["metadata_path"], "w", encoding="utf-8") as handle:
            json.dump(final_metadata, handle, indent=2)

        status = "complete" if completed else "partial"

        if not completed:
            logging.warning(
                f"⚠️ Partial processing for {video}/{plan['block_name']}: "
                f"processed {processed_count}/{planned_count} planned target frames."
            )

        block_summaries.append(
            {
                "Patient": patient,
                "Video": video,
                "Block_Name": plan["block_name"],
                "Class_Label": plan["class_label"],
                "Raw_Label": plan["raw_label"],
                "Face_Marked_Interval": bool(plan["face_marked_interval"]),
                "Occurrence_Index": plan["occurrence_index"],
                "Original_Start_Sec": round_float(plan["original_start"]),
                "Original_End_Sec": round_float(plan["original_end"]),
                "Buffered_Start_Sec": round_float(plan["buffered_start"]),
                "Buffered_End_Sec": round_float(plan["buffered_end"]),
                "Target_FPS": target_fps,
                "Boundary_Buffer_Sec": boundary_buffer_sec,
                "Timing_Mode": timing_mode,
                "Planned_Frames": planned_count,
                "Processed_Frames": processed_count,
                "Accepted_Frames": accepted_count,
                "Rejected_Frames": rejected_count,
                "Status": status,
            }
        )

    return (
        block_summaries,
        accepted_manifest_paths,
        rejected_manifest_paths,
        unreadable_videos,
    )


def append_dict_rows_to_csv(rows: list[dict[str, Any]], csv_path: str) -> None:
    """
    Append a small batch of dictionaries to a CSV without accumulating the full run in RAM.
    """
    if not rows:
        return

    csv_exists = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0
    fieldnames = list(rows[0].keys())

    with open(csv_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not csv_exists:
            writer.writeheader()
        writer.writerows(rows)


def append_manifest_csv_to_global(local_csv_path: str, global_csv_path: str) -> None:
    """
    Stream one per-block manifest into the global manifest CSV.
    """
    if not os.path.isfile(local_csv_path) or os.path.getsize(local_csv_path) == 0:
        return

    with open(local_csv_path, "r", newline="", encoding="utf-8") as source_handle:
        reader = csv.DictReader(source_handle)
        if not reader.fieldnames:
            return

        global_exists = os.path.isfile(global_csv_path) and os.path.getsize(global_csv_path) > 0

        with open(global_csv_path, "a", newline="", encoding="utf-8") as global_handle:
            writer = csv.DictWriter(global_handle, fieldnames=reader.fieldnames)
            if not global_exists:
                writer.writeheader()

            for row in reader:
                writer.writerow(row)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()

    if not validate_args(args):
        return

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "process.log")

    manager = Manager()
    log_queue = manager.Queue(-1)
    listener = start_log_listener(log_queue, log_path)
    configure_queue_logging(log_queue)

    try:
        logging.info("🚀 Starting publication-grade frame extraction.")
        logging.info(f"Excel file: {args.excel_path}")
        logging.info(f"Video directory: {args.video_dir}")
        logging.info(f"Output directory: {args.output_dir}")
        logging.info(f"Target FPS: {args.fps}")
        logging.info(f"Boundary buffer: {args.boundary_buffer_sec:.3f}s")
        logging.info(f"JPEG quality: {args.jpeg_quality}")
        logging.info(f"Force re-extract: {args.force_reextract}")
        logging.info(f"Quality filtering enabled: {args.enable_quality_filter}")
        logging.info(f"Face filtering enabled: {args.enable_face_filter}")
        logging.info(f"Face filtering scope: {args.face_filter_scope}")
        logging.info(f"Timing mode: {args.timing_mode}")
        logging.info(f"Two-field time interpretation: {args.two_field_time_format}")
        logging.info(
            "Decode mode: cap.grab() for sequential advancement, "
            "cap.retrieve() only for frames matching extraction targets."
        )

        try:
            raw_df = pd.read_excel(args.excel_path)
            df = standardize_excel_columns(raw_df)
        except Exception as exc:
            logging.error(f"❌ Failed to read or validate Excel file: {exc}")
            return

        timeline = build_timeline(
            df,
            two_field_time_format=args.two_field_time_format,
        )

        if not timeline:
            logging.warning("⚠️ No valid timeline entries were generated. Stopping.")
            return

        config: dict[str, Any] = {
            "target_fps": float(args.fps),
            "boundary_buffer_sec": float(args.boundary_buffer_sec),
            "jpeg_quality": int(args.jpeg_quality),
            "force_reextract": bool(args.force_reextract),
            "enable_quality_filter": bool(args.enable_quality_filter),
            "min_mean_intensity": float(args.min_mean_intensity),
            "max_mean_intensity": float(args.max_mean_intensity),
            "min_laplacian_var": float(args.min_laplacian_var),
            "enable_face_filter": bool(args.enable_face_filter),
            "face_filter_scope": str(args.face_filter_scope),
            "frontal_cascade_path": str(args.frontal_cascade_path),
            "profile_cascade_path": str(args.profile_cascade_path),
            "face_scale_factor": float(args.face_scale_factor),
            "face_min_neighbors": int(args.face_min_neighbors),
            "face_min_size": int(args.face_min_size),
            "timing_mode": str(args.timing_mode),
        }

        tasks = discover_video_tasks(
            timeline=timeline,
            video_dir=args.video_dir,
            output_dir=args.output_dir,
            config=config,
        )

        if not tasks:
            logging.warning("⚠️ No matched video tasks found. Stopping.")
            return

        if args.workers > 0:
            workers = min(args.workers, len(tasks))
        else:
            workers = min(cpu_count(), len(tasks))

        logging.info(f"🔧 Using {workers} worker processes.")

        global_block_summary_path = os.path.join(
            args.output_dir,
            "global_block_summary.csv",
        )
        global_accepted_manifest_path = os.path.join(
            args.output_dir,
            "global_accepted_frame_manifest.csv",
        )
        global_rejected_manifest_path = os.path.join(
            args.output_dir,
            "global_rejected_frame_manifest.csv",
        )

        # Rebuild global CSVs from this run, even if blocks are resumed.
        for global_path in (
            global_block_summary_path,
            global_accepted_manifest_path,
            global_rejected_manifest_path,
        ):
            if os.path.exists(global_path):
                os.remove(global_path)

        all_block_summaries: list[dict[str, Any]] = []
        unreadable_videos: list[str] = []

        with Pool(
            processes=workers,
            initializer=init_worker_logging,
            initargs=(log_queue,),
        ) as pool:
            for (
                block_summaries,
                accepted_manifest_paths,
                rejected_manifest_paths,
                unreadable,
            ) in pool.imap_unordered(process_video, tasks, chunksize=1):
                all_block_summaries.extend(block_summaries)
                unreadable_videos.extend(unreadable)

                append_dict_rows_to_csv(
                    block_summaries,
                    global_block_summary_path,
                )

                for local_manifest in accepted_manifest_paths:
                    append_manifest_csv_to_global(
                        local_manifest,
                        global_accepted_manifest_path,
                    )

                for local_manifest in rejected_manifest_paths:
                    append_manifest_csv_to_global(
                        local_manifest,
                        global_rejected_manifest_path,
                    )

        logging.info(f"📄 Saved global block summary: {global_block_summary_path}")
        logging.info(f"📄 Saved accepted-frame manifest: {global_accepted_manifest_path}")
        logging.info(f"📄 Saved rejected-frame manifest: {global_rejected_manifest_path}")

        if all_block_summaries:
            summary_df = pd.DataFrame(all_block_summaries)

            for patient, patient_df in summary_df.groupby("Patient"):
                patient_dir = os.path.join(args.output_dir, patient)
                os.makedirs(patient_dir, exist_ok=True)

                patient_summary_path = os.path.join(
                    patient_dir,
                    f"{patient}_block_summary.csv",
                )

                patient_df.to_csv(patient_summary_path, index=False)

        if unreadable_videos:
            unreadable_path = os.path.join(
                args.output_dir,
                "missing_or_unreadable_videos.txt",
            )

            with open(unreadable_path, "w", encoding="utf-8") as handle:
                handle.write("\n".join(sorted(set(unreadable_videos))))

            logging.warning(
                f"⚠️ {len(set(unreadable_videos))} videos were unreadable or failed "
                f"during decoding. Saved to {unreadable_path}"
            )

        logging.info("✅ Frame extraction completed.")

    finally:
        listener.stop()
        manager.shutdown()


if __name__ == "__main__":
    main()
