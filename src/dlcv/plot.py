import argparse
import pandas as pd
import matplotlib.pyplot as plt

from dlcv.config import get_cfg_defaults, CN

def plot_train_losses(csv_path, output_path):
    df = pd.read_csv(csv_path)
    plt.plot(df['Epoch'], df['Train Loss'], label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.legend()
    plt.savefig(output_path)
    plt.show()

def main(cfg):
    csv_path = cfg.TRAIN.RESULTS_CSV + "/" + cfg.TRAIN.RUN_NAME + ".csv"
    output_path = cfg.TRAIN.RESULTS_CSV + "/training_loss_plot.png"
    plot_train_losses(csv_path, output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot training losses.')
    parser.add_argument('--config', type=str, help='Path to config file', default=None)
    parser.add_argument('--opts', nargs='*', help='Modify config options using the command line')

    args = parser.parse_args()

    if args.config:
        cfg = CN.load_cfg(open(args.config, 'r'))
    else:
        cfg = get_cfg_defaults()

    if args.opts:
        for k, v in zip(args.opts[0::2], args.opts[1::2]):
            key_list = k.split('.')
            d = cfg
            for key in key_list[:-1]:
                d = d[key]
            d[key_list[-1]] = eval(v)

    main(cfg)
