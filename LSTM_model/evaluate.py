import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch import device
from lstm import SmileAuthenticityPredictor


def evaluate_data(
    model: SmileAuthenticityPredictor,
    device: device,
    angles: pd.DataFrame | list,
) -> float:
    """Evaluates smile authenticity percentage of given angles data.

    :param model: trained model object
    :type model: SmileAuthenticityPredictor
    :param device: device (CPU or GPU)
    :type device: device
    :param angles: angles data values
    :type angles: pd.DataFrame | list
    :return: autheniticity percentage
    :rtype: float
    """
    if isinstance(angles, pd.DataFrame):
        angles = angles.to_numpy()
    elif isinstance(angles, list):
        angles = np.array(angles)

    features_tensor = (
        torch.tensor(angles, dtype=torch.float32).transpose(0, 1).to(device)
    )
    _, output = model(features_tensor)
    probabilities = F.softmax(output, dim=1)
    authentic_smile_prob = probabilities[0][1].item()
    authentic_smile_percentage = authentic_smile_prob * 100
    return authentic_smile_percentage
