import subprocess
import time

def run_scenario(name, max_iters=2000, n_layer=4, n_head=4, batch_size=16):
    print(f"\n{'='*50}")
    print(f"Starting {name}...")
    print(f"{'='*50}")
    
    # building terminal command for train.py
    cmd = [
        "python", "src/train.py",
        "--scenario_name", name,
        "--max_iters", str(max_iters),
        "--n_layer", str(n_layer),
        "--n_head", str(n_head),
        "--batch_size", str(batch_size)
    ]

    # run the command
    start_time = time.time()
    subprocess.run(cmd, check=True)
    end_time = time.time()
    
    print(f"Finished {name} in {end_time - start_time:.1f} seconds.\n")

def main():
    print("Starting all experiments...")
    time.sleep(3) 

    # --- THE EXPERIMENTS ---
    
    try:
        # 0. Baseline
        run_scenario(name="baseline")
        
        # 1. Scenario 1: Varying training duration 
        run_scenario(name="scenario_1_short", max_iters=1000)
        run_scenario(name="scenario_1_long", max_iters=3000)

        # 2. Scenario 2: Model scaling
        run_scenario(name="scenario_2_scaled_arch", n_layer=8, n_head=8)
        
        # 3. Scenario 3: Varying batch size
        run_scenario(name="scenario_3_batch_8", batch_size=8)
        run_scenario(name="scenario_3_batch_32", batch_size=32)

        print("\nALL EXPERIMENTS COMPLETED!")
    except subprocess.CalledProcessError as e:
        print("\nEXPERIMENT RUN FAILED.")
        print(f"Command: {' '.join(e.cmd)}")
        print(f"Exit code: {e.returncode}")
        raise


if __name__ == "__main__":
    main()
