ntu_pairs = (
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
    (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
    (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
    (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),(25, 12)
)

LOWER_BODY_INDEX = [
    # 9, 10,      # left_hip, right_hip
    11, 12,     # left_knee, right_knee
    13, 14,     # left_ankle, right_ankle
    15, 16, 17, # left_big_toe, left_small_toe, left_heel
    18, 19, 20  # right_big_toe, right_small_toe, right_heel
]

from graph.mhr70_def import pose_info

# keypoint_info: {id: dict(name=..., id=..., ...)}
kinfo = pose_info["keypoint_info"]
name_to_id = {v["name"]: v["id"] for _, v in kinfo.items()}  # 0-based!!!!

# =============== full pairs (70 joints) ===============
mhr70_pairs = []
for _, info in pose_info["skeleton_info"].items():
    a, b = info["link"]
    mhr70_pairs.append((name_to_id[a], name_to_id[b]))
mhr70_pairs = tuple(mhr70_pairs)


# =============== upper pairs (old index space) ===============
remove_set = set(LOWER_BODY_INDEX)

mhr70_pairs_upper = []
for _, info in pose_info["skeleton_info"].items():
    a, b = info["link"]
    a_id = name_to_id[a]
    b_id = name_to_id[b]

    if a_id in remove_set or b_id in remove_set:
        continue

    mhr70_pairs_upper.append((a_id, b_id))

mhr70_pairs_upper = tuple(mhr70_pairs_upper)


# =============== build old->new mapping (re-index) ===============
keep_indices = [i for i in range(70) if i not in remove_set]  # old ids kept
old_to_new = {old_i: new_i for new_i, old_i in enumerate(keep_indices)}


# =============== renamed upper pairs (new index space) ===============
mhr70_pairs_upper_renamed = []
for (a_old, b_old) in mhr70_pairs_upper:
    a_new = old_to_new[a_old]
    b_new = old_to_new[b_old]
    mhr70_pairs_upper_renamed.append((a_new, b_new))

mhr70_pairs_upper = tuple(mhr70_pairs_upper_renamed)



