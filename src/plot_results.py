import pandas as pd
import matplotlib.pyplot as plt
import os

OUT_DIR = "out"
CSV_PATH = os.path.join(OUT_DIR, "emissions.csv")

def plot_results():
    if not os.path.exists(CSV_PATH):
        # Fallback hvis filen ligger i samme mappe
        df = pd.read_csv("emissions.csv")
    else:
        df = pd.read_csv(CSV_PATH)
        
    # Regn ut totalt strømtrekk (Watt) og gjør om energi til Watt-timer (Wh)
    df['total_power_w'] = df['cpu_power'] + df['gpu_power'] + df['ram_power']
    df['energy_wh'] = df['energy_consumed'] * 1000  # Omgjør fra kWh til Wh for finere tall
    
    # Klargjør figuren
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('LCA Scenario Analysis: Time, Energy, and Power Demand', fontsize=16, weight='bold')
    
    colors = ['#4A90E2', '#50E3C2', '#50E3C2', '#F5A623', '#E94A87', '#E94A87']
    labels = df['project_name'].tolist()
    
    # 1. Duration Plot
    axes[0].bar(labels, df['duration'], color=colors)
    axes[0].set_title('Training Duration (Seconds)')
    axes[0].set_ylabel('Seconds')
    axes[0].tick_params(axis='x', rotation=45)
    
    # 2. Energy Consumed Plot
    axes[1].bar(labels, df['energy_wh'], color=colors)
    axes[1].set_title('Total Energy Consumed (Wh)')
    axes[1].set_ylabel('Watt-hours (Wh)')
    axes[1].tick_params(axis='x', rotation=45)
    
    # 3. Average Power Demand Plot
    axes[2].bar(labels, df['total_power_w'], color=colors)
    axes[2].set_title('Average Power Demand (W)')
    axes[2].set_ylabel('Watts')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig("lca_scenario_results.png", dpi=300, bbox_inches='tight')
    print("Graf lagret som 'lca_scenario_results.png'")

if __name__ == "__main__":
    plot_results()