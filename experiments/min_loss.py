import pandas as pd
import argparse
import os


def min_loss(num):
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    metrics = os.path.abspath(os.path.join(ROOT_DIR, str(num), "epoch_metrics.csv"))
    data = pd.read_csv(metrics)
    val_data = data[data['phase'] == 'val']
    min_loss_row = val_data.loc[val_data['loss'].idxmin()]
    return min_loss_row['loss']

def main():
    parser = argparse.ArgumentParser(description="Uruchomienie testów i generowanie wykresów.")
    parser.add_argument(
        "num",
        metavar="N", 
        type=int, 
        nargs="+",
        help="Numer folderu (NUM), z którego mają być pobrane dane."
    )
    args = parser.parse_args()

    NUM = args.num

    for num in NUM:
        print(num, min_loss(num))


if __name__ == "__main__":
    main()