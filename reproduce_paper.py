import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Collect paper-grade results from experiment runs.")
    parser.add_argument("--runs", nargs="+", required=True, help="Experiment run directories.")
    parser.add_argument("--output-dir", default="paper_outputs", help="Directory for paper tables/results.")
    return parser.parse_args()


def main():
    from predictive_vision.reproduction import run_reproduction

    args = parse_args()
    run_reproduction(args.runs, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
