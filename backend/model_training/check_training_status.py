import os
import glob

def check_training_status():
    """
    Check if training has completed and show available models
    """
    print("Checking training status...")
    
    # Check for model files
    model_files = glob.glob("models/*.keras")
    model_files.extend(glob.glob("models/*.h5"))
    
    print(f"Found {len(model_files)} model file(s):")
    for model_file in model_files:
        file_size = os.path.getsize(model_file) / (1024 * 1024)  # Size in MB
        print(f"  - {model_file} ({file_size:.1f} MB)")
    
    # Check for results
    if os.path.exists("results"):
        result_files = os.listdir("results")
        print(f"\nFound {len(result_files)} result file(s):")
        for result_file in result_files:
            print(f"  - {result_file}")
    else:
        print("\nNo results directory found.")
    
    # Check for training logs
    log_files = glob.glob("*.log")
    if log_files:
        print(f"\nFound {len(log_files)} log file(s):")
        for log_file in log_files:
            print(f"  - {log_file}")
    
    return len(model_files) > 0

def main():
    """
    Main function to check training status
    """
    if check_training_status():
        print("\n✅ Training appears to be complete or in progress!")
        print("You can now run: python evaluate_model.py")
    else:
        print("\n❌ No trained model found.")
        print("Training may still be in progress or failed.")
        print("You can check the training by running: python train_model.py")

if __name__ == "__main__":
    main()
