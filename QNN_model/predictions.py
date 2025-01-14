import torch
import seaborn as sns
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from QNN_model.dataloader import FacesFeaturesDataset
from QNN_model.model import QuantumSmileClassifier
CLASSES_NAMES = CLASSES_STRS = ["0","1"]

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def show_conf_matrix(num, conf_matrix: np.ndarray) -> None:
    """Creates confusion matrix plot

    :param conf_matrix: confustion matrix array
    :type conf_matrix: np.ndarray
    """
    class_labels = ["Sztuczny", "Autentyczny"]
    hmap = sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    hmap.yaxis.set_ticklabels(
        hmap.yaxis.get_ticklabels(), rotation=0, ha="right"
    )
    hmap.xaxis.set_ticklabels(
        hmap.xaxis.get_ticklabels(), rotation=0, ha="right"
    )
    plt.title("Macierz błędów")
    plt.ylabel("Rzeczywiste")
    plt.xlabel("Przewidywane")
    plt.tight_layout()
    plt.savefig(os.path.abspath(os.path.join(os.sep, ROOT_DIR, "experiments", str(num), "confusion_matrix.png")))


def review_predictions(test_data: pd.DataFrame, num, ckpt_path: str) -> None:
    """Reviews predictions for test data given the checkpoint path.

    :param test_data: test data
    :type test_data: pd.DataFrame
    :param ckpt_path: model checkpoint path
    :type ckpt_path: str
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trained_model = QuantumSmileClassifier.load_from_checkpoint(
        ckpt_path, num_features=10
    )
    trained_model.to(device)
    trained_model.freeze()
    trained_model.eval()

    test_dataset = FacesFeaturesDataset(test_data)
    predictions, auths = [], []

    for item in tqdm(test_dataset):
        ffs = item["faces_features"].to(device)
        auth = item["authenticity"]

        output = trained_model(ffs.unsqueeze(dim=0))
        prediction = torch.argmax(output, dim=1)
        predictions.append(prediction.item())
        auths.append(auth.item())
    print(predictions)
    print(auths)
    print(classification_report(auths, predictions, target_names=CLASSES_STRS))

    cm = confusion_matrix(auths, predictions)
    print(cm)
    cm_df = pd.DataFrame(cm, index=CLASSES_NAMES, columns=CLASSES_NAMES)

    show_conf_matrix(num, cm_df)

def review_predictions2(test_data: pd.DataFrame, num, ckpt_path: str) -> None:
    """Reviews predictions for test data given the checkpoint path.

    :param test_data: test data
    :type test_data: pd.DataFrame
    :param ckpt_path: model checkpoint path
    :type ckpt_path: str
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trained_model = QuantumSmileClassifier.load_from_checkpoint(
        ckpt_path, num_features=10
    )
    trained_model.to(device)
    trained_model.freeze()
    trained_model.eval()

    test_dataset = FacesFeaturesDataset(test_data)
    predictions, auths = [], []

    for item in tqdm(test_dataset):
        ffs = item["faces_features"].to(device)
        auth = item["authenticity"]

        output = trained_model(ffs.unsqueeze(dim=0))
        prediction = torch.argmax(output, dim=1)
        predictions.append(prediction.item())
        auths.append(auth.item())
    return accuracy_score(auths, predictions)
