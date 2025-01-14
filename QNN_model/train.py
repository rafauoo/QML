import torch
from sklearn.model_selection import train_test_split
from QNN_model.trainer import get_trainer
from QNN_model.model import QuantumSmileClassifier
from QNN_model.predictions import review_predictions
from QNN_model.data import load_data_from_csv
from QNN_model.dataloader import FacesFeaturesDataModule

if __name__ == "__main__":
    data = load_data_from_csv("out")
    train_data, test_data = train_test_split(
        data, test_size=0.2, random_state=987876
    )
    batch_size = 64
    data_module = FacesFeaturesDataModule(
        train_data, test_data, batch_size=batch_size
    )
    data_module.setup()
    model = model = QuantumSmileClassifier(num_features=10)
    trainer = get_trainer()
    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module, ckpt_path="best")
    review_predictions(
        test_data, trainer.checkpoint_callback.best_model_path
    )