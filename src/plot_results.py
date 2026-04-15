"""Create Task 2 plots for training (scenarios 1-3) and inference (scenarios 4-5).

Outputs:
- task2_results.png: combined training + inference figure
- task2_summary.csv: training scenario metrics and deltas vs baseline
- task2_inference_summary.csv: aggregated inference metrics by setting
"""

import argparse
import csv
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

SCENARIOS = [
    "baseline",
    "scenario_1_short",
    "scenario_1_long",
    "scenario_2_scaled_arch",
    "scenario_3_batch_8",
    "scenario_3_batch_32",
]

SCENARIO_LABELS = {
    "baseline": "Baseline",
    "scenario_1_short": "S1 short",
    "scenario_1_long": "S1 long",
    "scenario_2_scaled_arch": "S2 scaled",
    "scenario_3_batch_8": "S3 batch=8",
    "scenario_3_batch_32": "S3 batch=32",
}

SCENARIO_FAMILY = {
    "baseline": "baseline",
    "scenario_1_short": "scenario_1",
    "scenario_1_long": "scenario_1",
    "scenario_2_scaled_arch": "scenario_2",
    "scenario_3_batch_8": "scenario_3",
    "scenario_3_batch_32": "scenario_3",
}

FAMILY_COLORS = {
    "baseline": "#4C78A8",
    "scenario_1": "#F58518",
    "scenario_2": "#54A24B",
    "scenario_3": "#E45756",
}


@dataclass
class ScenarioMetrics:
    scenario: str
    emissions_kg: float
    final_val_loss: float


@dataclass
class InferenceAggregate:
    scenario: str
    x_value: int
    mean_emissions_kg: float
    std_emissions_kg: float
    mean_elapsed_s: float
    std_elapsed_s: float
    runs: int


def load_latest_emissions(emissions_csv: str) -> Dict[str, float]:
    latest: Dict[str, tuple] = {}

    with open(emissions_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scenario = row.get("project_name", "")
            if scenario not in SCENARIOS:
                continue

            ts = datetime.fromisoformat(row["timestamp"])
            emissions = float(row["emissions"])
            prev = latest.get(scenario)
            if prev is None or ts > prev[0]:
                latest[scenario] = (ts, emissions)

    return {scenario: values[1] for scenario, values in latest.items()}


def load_loss_curves(out_dir: str) -> Dict[str, List[dict]]:
    curves = {}
    for scenario in SCENARIOS:
        path = os.path.join(out_dir, f"{scenario}_loss.txt")
        if not os.path.exists(path):
            continue
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            if rows:
                curves[scenario] = [
                    {
                        "iter": int(r["iter"]),
                        "train_loss": float(r["train_loss"]),
                        "val_loss": float(r["val_loss"]),
                    }
                    for r in rows
                ]
    return curves


def build_metrics(emissions: Dict[str, float], curves: Dict[str, List[dict]]) -> List[ScenarioMetrics]:
    metrics: List[ScenarioMetrics] = []
    for scenario in SCENARIOS:
        if scenario not in emissions or scenario not in curves:
            continue
        metrics.append(
            ScenarioMetrics(
                scenario=scenario,
                emissions_kg=emissions[scenario],
                final_val_loss=curves[scenario][-1]["val_loss"],
            )
        )
    return metrics


def load_inference_rows(inference_csv: str) -> List[dict]:
    if not os.path.exists(inference_csv):
        raise FileNotFoundError(
            f"Inference results not found: {inference_csv}. "
            "Run src/run_inference_experiments.py first."
        )
    with open(inference_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"Inference CSV is empty: {inference_csv}")
    return rows


def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, var ** 0.5


def aggregate_inference(rows: List[dict], scenario_name: str, x_field: str) -> List[InferenceAggregate]:
    grouped_emissions: Dict[int, List[float]] = defaultdict(list)
    grouped_elapsed: Dict[int, List[float]] = defaultdict(list)

    for row in rows:
        if row.get("scenario") != scenario_name:
            continue
        x_val = int(row[x_field])
        grouped_emissions[x_val].append(float(row["emissions_kg_co2eq"]))
        grouped_elapsed[x_val].append(float(row["elapsed_s"]))

    if not grouped_emissions:
        raise RuntimeError(
            f"No rows for '{scenario_name}' in inference CSV. "
            "Check output from run_inference_experiments.py."
        )

    aggregates: List[InferenceAggregate] = []
    for x_val in sorted(grouped_emissions):
        em_mean, em_std = _mean_std(grouped_emissions[x_val])
        t_mean, t_std = _mean_std(grouped_elapsed[x_val])
        aggregates.append(
            InferenceAggregate(
                scenario=scenario_name,
                x_value=x_val,
                mean_emissions_kg=em_mean,
                std_emissions_kg=em_std,
                mean_elapsed_s=t_mean,
                std_elapsed_s=t_std,
                runs=len(grouped_emissions[x_val]),
            )
        )
    return aggregates


def write_summary(metrics: List[ScenarioMetrics], output_csv: str) -> None:
    baseline = next((m for m in metrics if m.scenario == "baseline"), None)
    if baseline is None:
        raise ValueError("Baseline is required to compute relative deltas.")

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "scenario",
                "label",
                "family",
                "emissions_kg_co2eq",
                "final_val_loss",
                "delta_emissions_vs_baseline_pct",
                "delta_val_loss_vs_baseline_pct",
            ]
        )
        for m in metrics:
            d_em = 100.0 * (m.emissions_kg - baseline.emissions_kg) / baseline.emissions_kg
            d_loss = 100.0 * (m.final_val_loss - baseline.final_val_loss) / baseline.final_val_loss
            writer.writerow(
                [
                    m.scenario,
                    SCENARIO_LABELS[m.scenario],
                    SCENARIO_FAMILY[m.scenario],
                    f"{m.emissions_kg:.10f}",
                    f"{m.final_val_loss:.6f}",
                    f"{d_em:.2f}",
                    f"{d_loss:.2f}",
                ]
            )


def write_inference_summary(
    scenario4: List[InferenceAggregate], scenario5: List[InferenceAggregate], output_csv: str
) -> None:
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "scenario",
                "x_name",
                "x_value",
                "runs",
                "mean_emissions_kg_co2eq",
                "std_emissions_kg_co2eq",
                "mean_elapsed_s",
                "std_elapsed_s",
            ]
        )
        for row in scenario4:
            writer.writerow(
                [
                    row.scenario,
                    "prompt_chars",
                    row.x_value,
                    row.runs,
                    f"{row.mean_emissions_kg:.10f}",
                    f"{row.std_emissions_kg:.10f}",
                    f"{row.mean_elapsed_s:.6f}",
                    f"{row.std_elapsed_s:.6f}",
                ]
            )
        for row in scenario5:
            writer.writerow(
                [
                    row.scenario,
                    "max_new_tokens",
                    row.x_value,
                    row.runs,
                    f"{row.mean_emissions_kg:.10f}",
                    f"{row.std_emissions_kg:.10f}",
                    f"{row.mean_elapsed_s:.6f}",
                    f"{row.std_elapsed_s:.6f}",
                ]
            )


def make_plot(
    curves: Dict[str, List[dict]],
    metrics: List[ScenarioMetrics],
    scenario4: List[InferenceAggregate],
    scenario5: List[InferenceAggregate],
    output_img: str,
) -> None:
    baseline = next((m for m in metrics if m.scenario == "baseline"), None)
    if baseline is None:
        raise ValueError("Baseline is required for plotting deltas.")

    order = [s for s in SCENARIOS if any(m.scenario == s for m in metrics)]
    metric_map = {m.scenario: m for m in metrics}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    fig.suptitle("Task 2 Summary: Training Trade-off and Inference Scaling", fontsize=15)

    # 1) Training trade-off (simple and report-friendly)
    ax = axes[0]
    for scenario in order:
        m = metric_map[scenario]
        family = SCENARIO_FAMILY[scenario]
        c = FAMILY_COLORS[family]
        marker_size = 95 if scenario == "baseline" else 70
        edge = "black" if scenario == "baseline" else "none"
        ax.scatter(m.emissions_kg, m.final_val_loss, color=c, s=marker_size, edgecolors=edge, linewidths=1.0)
        ax.annotate(
            SCENARIO_LABELS[scenario],
            (m.emissions_kg, m.final_val_loss),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_title("Training Trade-off")
    ax.set_xlabel("Emissions (kg CO2e)")
    ax.set_ylabel("Final validation loss")
    ax.grid(alpha=0.25)

    # 2) Relative training change vs baseline (%)
    ax = axes[1]
    delta_emissions = []
    labels_no_baseline = []
    for scenario in order:
        if scenario == "baseline":
            continue
        m = metric_map[scenario]
        labels_no_baseline.append(SCENARIO_LABELS[scenario])
        delta_emissions.append(100.0 * (m.emissions_kg - baseline.emissions_kg) / baseline.emissions_kg)

    x = range(len(labels_no_baseline))
    ax.bar(list(x), delta_emissions, label="CO2e change %", color="#4C78A8")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels_no_baseline, rotation=35)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Training vs Baseline (%)")
    ax.set_ylabel("CO2e change (%)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)

    # 3) Inference scaling (S4 and S5), emissions only
    ax = axes[2]
    x4 = [r.x_value for r in scenario4]
    em4_mg = [r.mean_emissions_kg * 1_000_000 for r in scenario4]
    x5 = [r.x_value for r in scenario5]
    em5_mg = [r.mean_emissions_kg * 1_000_000 for r in scenario5]
    pos4 = list(range(len(x4)))
    gap = len(x4) + 2
    pos5 = [gap + i for i in range(len(x5))]

    ax.bar(pos4, em4_mg, color="#72B7B2", alpha=0.9, label="S4 emissions")
    ax.bar(pos5, em5_mg, color="#F58518", alpha=0.9, label="S5 emissions")
    ax.set_title("Inference Scaling (CO2e only)")
    ax.set_ylabel("Mean emissions (mg CO2e)")
    ax.grid(axis="y", alpha=0.25)

    tick_positions = pos4 + pos5
    tick_labels = [f"S4:{v}" for v in x4] + [f"S5:{v}" for v in x5]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=30)

    ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_img) or ".", exist_ok=True)
    plt.savefig(output_img, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot Task 2 training + inference scenario results")
    parser.add_argument("--out_dir", default="out", help="Directory with result files")
    parser.add_argument("--emissions_csv", default=os.path.join("out", "emissions.csv"))
    parser.add_argument(
        "--inference_csv",
        default=os.path.join("out", "inference_scenarios_results.csv"),
        help="CSV from run_inference_experiments.py",
    )
    parser.add_argument("--output_img", default="task2_results.png")
    parser.add_argument("--output_csv", default=os.path.join("out", "task2_summary.csv"), help="Training summary CSV")
    parser.add_argument(
        "--output_inference_csv",
        default=os.path.join("out", "task2_inference_summary.csv"),
        help="Inference aggregated summary CSV",
    )
    args = parser.parse_args()

    emissions = load_latest_emissions(args.emissions_csv)
    curves = load_loss_curves(args.out_dir)
    metrics = build_metrics(emissions, curves)
    inference_rows = load_inference_rows(args.inference_csv)
    scenario4 = aggregate_inference(inference_rows, "scenario_4_prompt_length", "prompt_chars")
    scenario5 = aggregate_inference(inference_rows, "scenario_5_output_length", "max_new_tokens")

    if not metrics:
        raise RuntimeError("No matching scenario data found. Check out/emissions.csv and out/*_loss.txt")

    write_summary(metrics, args.output_csv)
    write_inference_summary(scenario4, scenario5, args.output_inference_csv)
    make_plot(curves, metrics, scenario4, scenario5, args.output_img)

    print(f"Saved plot to: {args.output_img}")
    print(f"Saved training summary to: {args.output_csv}")
    print(f"Saved inference summary to: {args.output_inference_csv}")


if __name__ == "__main__":
    main()
