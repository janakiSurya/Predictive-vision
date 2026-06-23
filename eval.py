import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a predictive vision checkpoint.")
    parser.add_argument("--config", required=True, help="Path to an evaluation YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt or last.pt.")
    return parser.parse_args()


def main():
    from predictive_vision.evaluation import run_evaluation

    args = parse_args()
    run_evaluation(args.config, args.checkpoint)


if __name__ == "__main__":
    main()
