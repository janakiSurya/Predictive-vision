import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Generate predictive vision paper figures.")
    parser.add_argument("--run", required=True, help="Experiment run directory.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to visualize.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--sample-index", type=int, default=0)
    return parser.parse_args()


def main():
    from predictive_vision.plotting import run_plotting

    args = parse_args()
    run_plotting(args.run, args.checkpoint, split=args.split, sample_index=args.sample_index)


if __name__ == "__main__":
    main()
