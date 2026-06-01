import sys

from horse_behavior import infer_behavior_lightgbm
from horse_behavior import infer_behavior_roi_rules


def _parse_with_method_parser(argv: list[str] | None):
    argv = list(argv or sys.argv[1:])
    method = "lightgbm"
    if "--method" in argv:
        index = argv.index("--method")
        if index + 1 >= len(argv):
            raise SystemExit("--method requires a value")
        method = argv[index + 1]
        del argv[index : index + 2]
    elif argv and argv[0] in {"lightgbm", "roi-rules"}:
        method = argv.pop(0)

    parsers = {
        "lightgbm": infer_behavior_lightgbm.build_parser,
        "roi-rules": infer_behavior_roi_rules.build_parser,
    }
    if method not in parsers:
        raise SystemExit(f"Unsupported method: {method}")
    return method, parsers[method]().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    method, args = _parse_with_method_parser(argv)
    if method == "lightgbm":
        return infer_behavior_lightgbm.run(args)
    if method == "roi-rules":
        return infer_behavior_roi_rules.run(args)
    raise SystemExit(f"Unsupported method: {method}")


if __name__ == "__main__":
    raise SystemExit(main())
