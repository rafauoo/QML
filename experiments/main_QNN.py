import argparse
from QNN_model.test import test
from experiments.acc_plot import save_acc_plot
from experiments.loss_plot import save_loss_plot


def main():
    parser = argparse.ArgumentParser(description="Uruchomienie testów i generowanie wykresów.")
    parser.add_argument(
        "num",
        type=int, 
        help="Numer folderu (NUM), z którego mają być pobrane dane."
    )
    args = parser.parse_args()

    NUM = args.num

    test(NUM)
    save_acc_plot(NUM)
    save_loss_plot(NUM)


if __name__ == "__main__":
    main()
