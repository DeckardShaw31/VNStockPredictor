import subprocess
import sys

def main():
    print("Step 1: Fetching data...")
    subprocess.run([sys.executable, "src/data_fetcher.py"], check=True)

    print("\nStep 2: Computing features...")
    subprocess.run([sys.executable, "src/features.py"], check=True)

    print("\nStep 3: Generating signals...")
    subprocess.run([sys.executable, "src/strategies.py"], check=True)

    print("\nPipeline complete! Output saved to data/liveData.json")

if __name__ == "__main__":
    main()
