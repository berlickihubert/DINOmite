import re
import os
from plot_loss_acc import plot_acc, plot_loss
import matplotlib.pyplot as plt

def extract_accuracies(file_path):
    rob_acc_list = []
    nat_acc_list = []
    
    # Regex pattern to match rob_acc and nat_acc
    # Example: rob_acc=29.4, nat_acc=52.6
    pattern = re.compile(r"rob_acc=([\d\.]+),\s*nat_acc=([\d\.]+)")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    rob_acc = float(match.group(1))
                    nat_acc = float(match.group(2))
                    
                    rob_acc_list.append(rob_acc)
                    nat_acc_list.append(nat_acc)
                    
        return rob_acc_list, nat_acc_list

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return [], []
    except Exception as e:
        print(f"An error occurred: {e}")
        return [], []

if __name__ == "__main__":
    file_path = "temp3.txt" 

    
    if not os.path.exists(file_path):
         file_path = r"c:\Users\berli\Desktop\projects\DINOmite\temp3.txt"

    rob_accs, nat_accs = extract_accuracies(file_path)
    
    print(f"Found {len(rob_accs)} entries.")
    print("Robust Accuracies:", rob_accs)
    print("Natural Accuracies:", nat_accs)

    plt.figure(figsize=(7, 5))
    plt.plot(rob_accs, label='Robust accuracy (%)', color='blue')
    plt.plot(nat_accs, label='Accuracy (%)', color='green')
    plt.title('Model accuracy over training steps for TRADES method')
    plt.xlabel('training steps')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    plt.grid(True)
    plt.legend()
    plt.savefig('adversarial_examples/accuracy_plot_trades.png')
    plt.close()
