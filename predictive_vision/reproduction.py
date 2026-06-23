import csv
import json
from pathlib import Path

from predictive_vision.config import save_json


def _load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def collect_run(run_dir):
    run_dir = Path(run_dir)
    summary_path = run_dir / "summary.json"
    eval_dir = run_dir / "eval"
    eval_summaries = sorted(eval_dir.glob("*_summary.json")) if eval_dir.exists() else []
    summary = _load_json(summary_path) if summary_path.exists() else {}
    eval_payloads = [_load_json(path) for path in eval_summaries]
    figures = sorted(str(path) for path in (run_dir / "figures").glob("*.png")) if (run_dir / "figures").exists() else []
    return {
        "run_dir": str(run_dir),
        "summary": summary,
        "eval": eval_payloads,
        "figures": figures,
    }


def run_reproduction(runs, output_dir="paper_outputs"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_payload = [collect_run(run) for run in runs]
    save_json({"runs": runs_payload}, output_dir / "paper_results.json")

    density_rows = []
    ablation_rows = []
    for payload in runs_payload:
        run_name = Path(payload["run_dir"]).name
        best = payload["summary"].get("best_metrics") or payload["summary"].get("last_validation") or {}
        density_rows.append(
            {
                "run": run_name,
                "raw_density": best.get("raw_density"),
                "cleaned_density": best.get("cleaned_density"),
                "residual_density": best.get("residual_density"),
                "sparse_density": best.get("sparse_density"),
                "data_reduction": best.get("data_reduction"),
            }
        )
        ablation_rows.append(
            {
                "run": run_name,
                "raw_accuracy": best.get("raw_accuracy"),
                "sparse_accuracy": best.get("sparse_accuracy"),
                "data_reduction": best.get("data_reduction"),
                "beta_mean": best.get("beta_mean"),
                "regime_entropy": best.get("regime_entropy"),
                "episodic_rate": best.get("episodic_rate"),
            }
        )

    _write_csv(
        output_dir / "density_table.csv",
        density_rows,
        ["run", "raw_density", "cleaned_density", "residual_density", "sparse_density", "data_reduction"],
    )
    _write_csv(
        output_dir / "ablation_table.csv",
        ablation_rows,
        ["run", "raw_accuracy", "sparse_accuracy", "data_reduction", "beta_mean", "regime_entropy", "episodic_rate"],
    )
    print(f"Saved paper reproduction outputs to {output_dir}")
    return output_dir
