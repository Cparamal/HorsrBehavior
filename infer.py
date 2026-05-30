import sys

from horse_behavior import infer_behavior
from horse_behavior import infer_behavior_lightgbm
from horse_behavior import infer_behavior_pose_hybrid
from horse_behavior import infer_behavior_yolo_pose
from horse_behavior import infer_behavior_yolo_roi_cls

#.\.venv\Scripts\python.exe infer.py --method rules --source video/stable_20260522_155032.mp4 --output outputs/rules_video --max-frames 1800 --no-display
# .\.venv\Scripts\python.exe infer_roi_rules.py  --source video/stable_20260522_155032.mp4 --output outputs/roi_with_rulues__video.mp4 --max-frames 18000 --no-display

def _parse_with_method_parser(argv: list[str] | None):
    argv = list(argv or sys.argv[1:])
    method = "lightgbm"
    if "--method" in argv:
        index = argv.index("--method")
        if index + 1 >= len(argv):
            raise SystemExit("--method requires a value")
        method = argv[index + 1]
        del argv[index : index + 2]
    elif argv and argv[0] in {"rules", "lightgbm", "roi-yolo", "pose-yolo", "pose-hybrid"}:
        method = argv.pop(0)

    parsers = {
        "rules": infer_behavior.build_parser,
        "lightgbm": infer_behavior_lightgbm.build_parser,
        "roi-yolo": infer_behavior_yolo_roi_cls.build_parser,
        "pose-yolo": infer_behavior_yolo_pose.build_parser,
        "pose-hybrid": infer_behavior_pose_hybrid.build_parser,
    }
    if method not in parsers:
        raise SystemExit(f"Unsupported method: {method}")
    return method, parsers[method]().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    method, args = _parse_with_method_parser(argv)
    if method == "rules":
        return infer_behavior.main_from_args(args)
    if method == "lightgbm":
        return infer_behavior_lightgbm.run(args)
    if method == "roi-yolo":
        return infer_behavior_yolo_roi_cls.run(args)
    if method == "pose-yolo":
        return infer_behavior_yolo_pose.run(args)
    if method == "pose-hybrid":
        return infer_behavior_pose_hybrid.run(args)
    raise SystemExit(f"Unsupported method: {method}")


if __name__ == "__main__":
    raise SystemExit(main())
