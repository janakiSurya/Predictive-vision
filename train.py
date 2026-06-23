import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train the predictive vision paper pipeline.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--resume", default=None, help="Optional checkpoint path to resume.")
    return parser.parse_args()


def main():
    from predictive_vision.training import run_training

    args = parse_args()
    run_training(args.config, resume=args.resume)


if __name__ == "__main__":
    main()
