import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from src import config


def _text_or_none(node, names):
    for name in names:
        child = node.find(name)
        if child is not None and child.text is not None:
            return child.text.strip()
    return None


def _is_positive_event(name):
    if not name:
        return False
    parts = [part.strip().lower() for part in name.split("|")]
    return any(part in config.APNEA_EVENT_NAMES for part in parts)


def find_annotation_file(patient_id, annotations_dir=config.ANNOTATIONS_DIR):
    annotations_dir = Path(annotations_dir)
    candidates = sorted(annotations_dir.glob(f"{patient_id}*.xml"))
    candidates += sorted(annotations_dir.glob(f"{patient_id}*.sml"))
    return candidates[0] if candidates else None


def parse_apnea_events(annotation_path):
    """
    Parse positive apnea/hypopnea events from an NSRR XML/SML file.

    Returns a list of (start, end) intervals in seconds.
    """
    annotation_path = Path(annotation_path)
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    events = []
    for event in root.iter("ScoredEvent"):
        name = _text_or_none(event, ("EventConcept", "Name"))
        if not _is_positive_event(name):
            continue

        start_text = _text_or_none(event, ("Start",))
        duration_text = _text_or_none(event, ("Duration",))
        if start_text is None or duration_text is None:
            continue

        try:
            start = float(start_text)
            duration = float(duration_text)
        except ValueError:
            continue

        if duration <= 0:
            continue
        events.append((start, start + duration))

    return events


def load_apnea_events_for_patient(patient_id, annotations_dir=config.ANNOTATIONS_DIR):
    annotation_file = find_annotation_file(patient_id, annotations_dir)
    if annotation_file is None:
        print(f"Warning: no annotation file found for {patient_id}")
        return []

    try:
        return parse_apnea_events(annotation_file)
    except Exception as exc:
        print(f"Warning: could not parse annotations for {patient_id}: {exc}")
        return []


def label_windows(window_starts, window_ends, events, min_overlap_seconds=None):
    if min_overlap_seconds is None:
        min_overlap_seconds = config.MIN_APNEA_OVERLAP_SECONDS

    labels = []
    for start, end in zip(window_starts, window_ends):
        label = 0
        for event_start, event_end in events:
            overlap = min(end, event_end) - max(start, event_start)
            if overlap >= min_overlap_seconds:
                label = 1
                break
        labels.append(label)
    return np.asarray(labels, dtype=int)

