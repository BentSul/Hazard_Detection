
# hazard_scoring.py
#
# Takes YOLO detection results for a single frame/image and computes
# a composite hazard score: base severity per detected class, plus an
# escalation bonus when two dangerous classes are detected close together.

import math

# ---------------------------------------------------------------------
# 1. CONFIG — edit this as your class list grows
# ---------------------------------------------------------------------

# Base severity weight per class (0-10 scale, tune these as a team)
HAZARD_WEIGHTS = {
    "Overloaded_power_strip": 6,
    "Unstable_stack_of_books": 4,
    "Exposed_wires": 8,       # electrical + exposed = high severity, team's call
    "Spilled_liquid": 5,
    "Scattered_pile": 3,
    "loose_cable": 4,
    "Damaged_furniture": 5,
}

# Combo bonus — triggers when BOTH classes in the pair are detected
# close together in the same frame. Key = frozenset of the two class
# names (order doesn't matter), value = extra points added.
COMBO_BONUS = {
    frozenset(["Overloaded_power_strip", "Exposed_wires"]): 6,   # electrical hazards compounding
    frozenset(["Spilled_liquid", "Overloaded_power_strip"]): 8,   # liquid near electrical — dangerous combo
    frozenset(["Spilled_liquid", "Exposed_wires"]): 8,
    frozenset(["Unstable_stack_of_books", "Scattered_pile"]): 3,  # both trip/fall related
}

# How close two detections need to be (in pixels, based on box center
# distance) to count as "co-occurring" for a combo bonus. Tune this
# based on your image resolution / camera distance during testing.
PROXIMITY_THRESHOLD_PX = 250

# Score cutoffs for risk level labels
RISK_LEVELS = [
    (0, 5, "LOW"),
    (5, 12, "MODERATE"),
    (12, math.inf, "CRITICAL"),
]


# ---------------------------------------------------------------------
# 2. HELPERS
# ---------------------------------------------------------------------

def box_center(box_xyxy):
    """box_xyxy = [x1, y1, x2, y2] -> (center_x, center_y)"""
    x1, y1, x2, y2 = box_xyxy
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])    # calculating the euclidean distances from the centers of the bounding boxes


def risk_level(score):
    for low, high, label in RISK_LEVELS:
        if low <= score < high:
            return label
    return "UNKNOWN"


# ---------------------------------------------------------------------
# 3. CORE SCORING FUNCTION
# ---------------------------------------------------------------------

def score_detections(detections):
    """
    detections: list of dicts, one per detected object, e.g.
        [
            {"class_name": "overloaded_socket", "box": [x1, y1, x2, y2], "confidence": 0.91},
            {"class_name": "unstable_stacking", "box": [x1, y1, x2, y2], "confidence": 0.77},
        ]

    Returns a dict:
        {
            "base_score": float,
            "combo_bonus": float,
            "composite_score": float,
            "risk_level": str,
            "triggered_combos": list of (class_a, class_b) tuples
        }
    """
    base_score = 0.0
    for det in detections:
        base_score += HAZARD_WEIGHTS.get(det["class_name"], 0)

    combo_bonus = 0.0
    triggered_combos = []

    # check every pair of detections for a proximity-based combo
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            det_a, det_b = detections[i], detections[j]
            pair_key = frozenset([det_a["class_name"], det_b["class_name"]])

            if pair_key in COMBO_BONUS:
                center_a = box_center(det_a["box"])
                center_b = box_center(det_b["box"])
                if distance(center_a, center_b) <= PROXIMITY_THRESHOLD_PX:
                    combo_bonus += COMBO_BONUS[pair_key]
                    triggered_combos.append((det_a["class_name"], det_b["class_name"]))

    composite_score = base_score + combo_bonus

    return {
        "base_score": base_score,
        "combo_bonus": combo_bonus,
        "composite_score": composite_score,
        "risk_level": risk_level(composite_score),
        "triggered_combos": triggered_combos,
    }


# ---------------------------------------------------------------------
# 4. HELPER TO CONVERT YOLO/ultralytics RESULTS INTO THE FORMAT ABOVE
# ---------------------------------------------------------------------

def detections_from_yolo_result(result):
    """
    Converts an ultralytics `results[0]` object into the plain list-of-dicts
    format score_detections() expects.
    """
    detections = []
    names = result.names  # class index -> class name mapping
    for box in result.boxes:
        cls_idx = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
        detections.append({
            "class_name": names[cls_idx],
            "box": xyxy,
            "confidence": conf,
        })
    return detections


# ---------------------------------------------------------------------
# 5. QUICK TEST (run this file directly to sanity check the logic)
# ---------------------------------------------------------------------

# Setting the base condition for running the script
if __name__ == "__main__":
    # fake detections to prove the scoring logic works before touching YOLO
    fake_detections = [
        {"class_name": "Overloaded_power_strip", "box": [100, 100, 150, 150], "confidence": 0.9},
        {"class_name": "Unstable_stack_of_books", "box": [180, 120, 230, 170], "confidence": 0.8},
    ]  #Fake detections just to make sure what we did is consistent enough

    result = score_detections(fake_detections)
    print(result)
