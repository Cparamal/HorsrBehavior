SUPERANIMAL_QUADRUPED_KEYPOINTS = [
    "nose",
    "upper_jaw",
    "lower_jaw",
    "mouth_end_right",
    "mouth_end_left",
    "right_eye",
    "right_earbase",
    "right_earend",
    "right_antler_base",
    "right_antler_end",
    "left_eye",
    "left_earbase",
    "left_earend",
    "left_antler_base",
    "left_antler_end",
    "neck_base",
    "neck_end",
    "throat_base",
    "throat_end",
    "back_base",
    "back_end",
    "back_middle",
    "tail_base",
    "tail_end",
    "front_left_thai",
    "front_left_knee",
    "front_left_paw",
    "front_right_thai",
    "front_right_knee",
    "front_right_paw",
    "back_left_paw",
    "back_left_thai",
    "back_right_thai",
    "back_left_knee",
    "back_right_knee",
    "back_right_paw",
    "belly_bottom",
    "body_middle_right",
    "body_middle_left",
]

SUPERANIMAL_QUADRUPED_SKELETON = [
    ("nose", "upper_jaw"),
    ("nose", "lower_jaw"),
    ("upper_jaw", "right_eye"),
    ("upper_jaw", "left_eye"),
    ("right_eye", "right_earbase"),
    ("right_earbase", "right_earend"),
    ("left_eye", "left_earbase"),
    ("left_earbase", "left_earend"),
    ("upper_jaw", "neck_base"),
    ("lower_jaw", "throat_base"),
    ("neck_base", "neck_end"),
    ("throat_base", "throat_end"),
    ("neck_end", "back_base"),
    ("back_base", "back_middle"),
    ("back_middle", "back_end"),
    ("back_end", "tail_base"),
    ("tail_base", "tail_end"),
    ("neck_end", "front_left_thai"),
    ("front_left_thai", "front_left_knee"),
    ("front_left_knee", "front_left_paw"),
    ("neck_end", "front_right_thai"),
    ("front_right_thai", "front_right_knee"),
    ("front_right_knee", "front_right_paw"),
    ("back_end", "back_left_thai"),
    ("back_left_thai", "back_left_knee"),
    ("back_left_knee", "back_left_paw"),
    ("back_end", "back_right_thai"),
    ("back_right_thai", "back_right_knee"),
    ("back_right_knee", "back_right_paw"),
    ("belly_bottom", "body_middle_right"),
    ("belly_bottom", "body_middle_left"),
]

KEYPOINT_INDEX = {name: index for index, name in enumerate(SUPERANIMAL_QUADRUPED_KEYPOINTS)}
YOLO_POSE_CLASS_NAMES = {0: "horse"}

CORE_6_KEYPOINTS = [
    ("nose", "nose"),
    ("jaw", "upper_jaw"),
    ("withers", "neck_base"),
    ("neck_end", "neck_end"),
    ("mid_back", "back_middle"),
    ("croup", "back_end"),
]

CORE_6_SKELETON = [
    ("nose", "jaw"),
    ("nose", "neck_end"),
    ("withers", "neck_end"),
    ("neck_end", "mid_back"),
    ("mid_back", "croup"),
]


def skeleton_indices() -> list[list[int]]:
    return [[KEYPOINT_INDEX[a], KEYPOINT_INDEX[b]] for a, b in SUPERANIMAL_QUADRUPED_SKELETON]


def core_6_skeleton_indices() -> list[list[int]]:
    output_index = {name: index for index, (name, _source_name) in enumerate(CORE_6_KEYPOINTS)}
    return [[output_index[a], output_index[b]] for a, b in CORE_6_SKELETON]
