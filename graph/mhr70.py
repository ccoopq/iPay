import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools
from graph.mhr70_def import pose_info


LOWER_BODY_INDEX = [
    # 9, 10,      # left_hip, right_hip
    11, 12,     # left_knee, right_knee
    13, 14,     # left_ankle, right_ankle
    15, 16, 17, # left_big_toe, left_small_toe, left_heel
    18, 19, 20  # right_big_toe, right_small_toe, right_heel
]


def build_full_graph():
    """
    Build full-body graph with 70 joints (0-based).
    Return: num_node, self_link, inward, outward, neighbor
    """
    num_node = 70
    self_link = [(i, i) for i in range(num_node)]

    # keypoint_info: {idx: {"name": ..., "id": ...}}
    keypoint_info = pose_info["keypoint_info"]

    # STRICT 0-based mapping: name -> id (0~69)
    name_to_idx = {v["name"]: v["id"] for _, v in keypoint_info.items()}

    inward = []
    for _, info in pose_info["skeleton_info"].items():
        a_name, b_name = info["link"]

        if a_name not in name_to_idx or b_name not in name_to_idx:
            raise KeyError(f"Unknown joint name in skeleton_info: {a_name}, {b_name}")

        a = name_to_idx[a_name]
        b = name_to_idx[b_name]

        inward.append((a, b))

    # remove duplicates while preserving order
    _seen = set()
    inward_unique = []
    for e in inward:
        if e not in _seen:
            inward_unique.append(e)
            _seen.add(e)
    inward = inward_unique

    outward = [(j, i) for (i, j) in inward]
    neighbor = inward + outward

    return num_node, self_link, inward, outward, neighbor


def build_upper_graph():
    """
    Build upper-body graph by removing LOWER_BODY_INDEX joints.
    This re-indexes joints into a compact range [0, num_upper-1].
    """
    remove_set = set(LOWER_BODY_INDEX)

    # indices to keep (0-based)
    keep_indices = [i for i in range(70) if i not in remove_set]

    # old idx -> new idx
    old_to_new = {old_i: new_i for new_i, old_i in enumerate(keep_indices)}

    num_node = len(keep_indices)
    self_link = [(i, i) for i in range(num_node)]

    keypoint_info = pose_info["keypoint_info"]
    name_to_idx_full = {v["name"]: v["id"] for _, v in keypoint_info.items()}

    inward = []
    for _, info in pose_info["skeleton_info"].items():
        a_name, b_name = info["link"]

        if a_name not in name_to_idx_full or b_name not in name_to_idx_full:
            raise KeyError(f"Unknown joint name in skeleton_info: {a_name}, {b_name}")

        a_full = name_to_idx_full[a_name]
        b_full = name_to_idx_full[b_name]

        # skip edges involving removed lower-body joints
        if a_full in remove_set or b_full in remove_set:
            continue

        # remap to new compact indices
        a = old_to_new[a_full]
        b = old_to_new[b_full]
        inward.append((a, b))

    # remove duplicates while preserving order
    _seen = set()
    inward_unique = []
    for e in inward:
        if e not in _seen:
            inward_unique.append(e)
            _seen.add(e)
    inward = inward_unique

    outward = [(j, i) for (i, j) in inward]
    neighbor = inward + outward

    return num_node, self_link, inward, outward, neighbor


class Graph:
    def __init__(self, labeling_mode='spatial', upper=False):
        self.upper = upper

        if self.upper:
            num_node, self_link, inward, outward, neighbor = build_upper_graph()
        else:
            num_node, self_link, inward, outward, neighbor = build_full_graph()

        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A

        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(
                self.num_node,
                self.self_link,
                self.inward,
                self.outward
            )
        else:
            raise ValueError(f"Unsupported labeling_mode: {labeling_mode}")

        return A
