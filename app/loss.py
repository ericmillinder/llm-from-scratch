# pip install matplotlib
import matplotlib.pyplot as plt
from json import load


def plot_loss_curve(dir='.'):
    with open(f"{dir}/loss_log.json") as f:
        log = load(f)

    plt.figure(figsize=(10, 6))
    plt.title("Training Loss Curve")
    plt.plot(log["steps"], log["train"], alpha=0.3, label="train")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{dir}/loss_curve.png")
    plt.show()

if __name__ == "__main__":
    plot_loss_curve('tiny')