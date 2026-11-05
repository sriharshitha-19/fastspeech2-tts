import argparse

import yaml

from preprocessor.preprocessor import Preprocessor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=str,
        help="path to config.yaml (one file with sections: preprocess, model, train)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="smoke test: only process the first N clips of each split",
    )
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)["preprocess"]
    preprocessor = Preprocessor(config, limit=args.limit)
    preprocessor.build_from_path()
