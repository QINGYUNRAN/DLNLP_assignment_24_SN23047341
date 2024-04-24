import matplotlib.pyplot as plt
import os

dir_path = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(dir_path, '..', 'image_output')
def plot_loss(loss_histories):
    plt.figure(figsize=(12, 8))

    # Plot each fold's loss history with a label, including step counts
    for i, losses in enumerate(loss_histories):
        steps = range(0, len(losses) * 10, 10)  # Creating steps from 0, incrementing by 10
        plt.plot(steps, losses, label=f'fold{i}')

    # Adding legend, title, and labels
    plt.legend()
    plt.title('Loss History by Fold')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.tight_layout()
    # Display the plot
    plt.savefig(os.path.join(image_path, 'loss.pdf'))

def plot_f5(eval_f5_histories):
    plt.figure(figsize=(10, 5))
    for i, eval_f5s in enumerate(eval_f5_histories):
        if eval_f5s:  # Only plot if there are eval_f5 values
            steps = range(50, (len(eval_f5s) + 1) * 50, 50)
            plt.plot(steps, eval_f5s, label=f'fold{i}')
        max_value = max(eval_f5s)
        max_index = eval_f5s.index(max_value) + 1
        max_step = max_index * 50
        plt.annotate(f'{max_value:.3f}', (max_step, max_value), textcoords="offset points", xytext=(0, 10), ha='center')
        plt.scatter(max_step, max_value, s=50, zorder=5)  # 使用zorder确保点在最上层

    plt.legend()
    plt.title('Eval F5 by Fold')
    plt.xlabel('Steps')
    plt.ylabel('Eval F5')
    plt.savefig(os.path.join(image_path, 'f5.pdf'))