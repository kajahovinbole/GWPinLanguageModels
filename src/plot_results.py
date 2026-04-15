import argparse
import csv
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

# Dine nøyaktige filnavn
SCENARIOS = [
    "baseline",
    "scenario_1_1000",
    "scenario_1_1500",
    "scenario_1_2500",
    "scenario_1_3000",
    "scenario_2_scaled_1",
    "scenario_2_scaled_2",
    "scenario_2_scaled_3",
    "scenario_3_batch_8",
    "scenario_3_batch_12",
    "scenario_3_batch_24",
    "scenario_3_batch_32",
]

SCENARIO_LABELS = {
    "baseline": "Baseline",
    "scenario_1_1000": "S1 (1000)",
    "scenario_1_1500": "S1 (1500)",
    "scenario_1_2500": "S1 (2500)",
    "scenario_1_3000": "S1 (3000)",
    "scenario_2_scaled_1": "S2 Scaled 1",
    "scenario_2_scaled_2": "S2 Scaled 2",
    "scenario_2_scaled_3": "S2 Scaled 3",
    "scenario_3_batch_8": "S3 (B8)",
    "scenario_3_batch_12": "S3 (B12)",
    "scenario_3_batch_24": "S3 (B24)",
    "scenario_3_batch_32": "S3 (B32)",
}

SCENARIO_FAMILY = {
    "baseline": "baseline",
    "scenario_1_1000": "scenario_1",
    "scenario_1_1500": "scenario_1",
    "scenario_1_2500": "scenario_1",
    "scenario_1_3000": "scenario_1",
    "scenario_2_scaled_1": "scenario_2",
    "scenario_2_scaled_2": "scenario_2",
    "scenario_2_scaled_3": "scenario_2",
    "scenario_3_batch_8": "scenario_3",
    "scenario_3_batch_12": "scenario_3",
    "scenario_3_batch_24": "scenario_3",
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
    runs: int

def load_latest_emissions(emissions_csv: str) -> Dict[str, float]:
    latest = {}
    if not os.path.exists(emissions_csv): return latest
    with open(emissions_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scenario = row.get("project_name", "")
            if scenario not in SCENARIOS: continue
            ts = datetime.fromisoformat(row["timestamp"])
            emissions = float(row["emissions"])
            prev = latest.get(scenario)
            if prev is None or ts > prev[0]:
                latest[scenario] = (ts, emissions)
    return {k: v[1] for k, v in latest.items()}

def load_loss_curves(out_dir: str) -> Dict[str, List[dict]]:
    curves = {}
    for scenario in SCENARIOS:
        path = os.path.join(out_dir, f"{scenario}_loss.txt")
        if not os.path.exists(path): continue
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            if rows:
                curves[scenario] = [{"val_loss": float(r["val_loss"])} for r in rows]
    return curves

def load_inference_rows(inference_csv: str) -> List[dict]:
    if not os.path.exists(inference_csv): return []
    with open(inference_csv, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def aggregate_inference(rows: List[dict], scenario_name: str, x_field: str) -> List[InferenceAggregate]:
    grouped_emissions, grouped_elapsed = defaultdict(list), defaultdict(list)
    for row in rows:
        if row.get("scenario") != scenario_name: continue
        x_val = int(row[x_field])
        grouped_emissions[x_val].append(float(row["emissions_kg_co2eq"]))
        grouped_elapsed[x_val].append(float(row["elapsed_s"]))

    aggregates = []
    for x_val in sorted(grouped_emissions):
        em_vals, t_vals = grouped_emissions[x_val], grouped_elapsed[x_val]
        em_mean = sum(em_vals) / len(em_vals)
        t_mean = sum(t_vals) / len(t_vals)
        aggregates.append(InferenceAggregate(scenario_name, x_val, em_mean, 0.0, t_mean, len(em_vals)))
    return aggregates

def make_plot(curves, metrics, scenario4, scenario5, output_img):
    baseline = next((m for m in metrics if m.scenario == "baseline"), None)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Task 2 Summary: Training Trade-off and Inference Scaling", fontsize=16, fontweight='bold')

    # 1) Training trade-off
    ax = axes[0]
    metric_map = {m.scenario: m for m in metrics}
    for scenario in [s for s in SCENARIOS if s in metric_map]:
        m = metric_map[scenario]
        c = FAMILY_COLORS[SCENARIO_FAMILY[scenario]]
        size = 120 if scenario == "baseline" else 70
        edge = "black" if scenario == "baseline" else "none"
        ax.scatter(m.emissions_kg, m.final_val_loss, color=c, s=size, edgecolors=edge, linewidths=1.5)
        ax.annotate(SCENARIO_LABELS[scenario], (m.emissions_kg, m.final_val_loss), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_title("Training Trade-off (Energy vs Performance)")
    ax.set_xlabel("Emissions (kg CO2e)")
    ax.set_ylabel("Final Validation Loss")
    ax.grid(alpha=0.3)

    # 2) Relative training change
    ax = axes[1]
    if baseline:
        labels, deltas = [], []
        for scenario in [s for s in SCENARIOS if s in metric_map and s != "baseline"]:
            labels.append(SCENARIO_LABELS[scenario])
            deltas.append(100.0 * (metric_map[scenario].emissions_kg - baseline.emissions_kg) / baseline.emissions_kg)
        ax.bar(range(len(labels)), deltas, color="#4C78A8")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.axhline(0, color="black", linewidth=1.2)
        ax.set_title("Training Emissions vs Baseline (%)")
        ax.set_ylabel("CO2e change (%)")
        ax.grid(axis="y", alpha=0.3)

    # 3) Inference scaling
    ax = axes[2]
    if scenario4 and scenario5:
        x4 = [r.x_value for r in scenario4]
        em4 = [r.mean_emissions_kg * 1_000_000 for r in scenario4] # Konvertert til mg
        x5 = [r.x_value for r in scenario5]
        em5 = [r.mean_emissions_kg * 1_000_000 for r in scenario5]
        
        pos4 = list(range(len(x4)))
        pos5 = [len(x4) + 1 + i for i in range(len(x5))]

        ax.bar(pos4, em4, color="#72B7B2", alpha=0.9, label="S4 (Prompt Length)")
        ax.bar(pos5, em5, color="#F58518", alpha=0.9, label="S5 (Output Length)")
        ax.set_title("Inference Scaling")
        ax.set_ylabel("Mean emissions (mg CO2e)")
        ax.grid(axis="y", alpha=0.3)
        ax.set_xticks(pos4 + pos5)
        ax.set_xticklabels([f"In:{v}" for v in x4] + [f"Out:{v}" for v in x5], rotation=45, ha='right')
        ax.legend(fontsize=9, loc="upper left")
    else:
        ax.text(0.5, 0.5, "Ingen inferens-data funnet", ha='center', va='center')

    plt.tight_layout()
    plt.savefig(output_img, dpi=300, bbox_inches="tight")

def main():
    out_dir = "out"
    emissions_csv = os.path.join(out_dir, "emissions.csv")
    inference_csv = os.path.join(out_dir, "inference_scenarios_results.csv")
    output_img = os.path.join(out_dir, "task2_results.png")

    emissions = load_latest_emissions(emissions_csv)
    curves = load_loss_curves(out_dir)
    metrics = [ScenarioMetrics(s, emissions[s], curves[s][-1]["val_loss"]) for s in SCENARIOS if s in emissions and s in curves]
    
    inference_rows = load_inference_rows(inference_csv)
    scenario4 = aggregate_inference(inference_rows, "scenario_4_prompt_length", "prompt_chars")
    scenario5 = aggregate_inference(inference_rows, "scenario_5_output_length", "max_new_tokens")

    make_plot(curves, metrics, scenario4, scenario5, output_img)
    print(f"\n🎉 SUCCESS! Du finner den ferdige grafen i: {output_img}")

if __name__ == "__main__":
    main()