from LSTM_model.lstm import SmileAuthenticityPredictor
from LSTM_model.data import load_data_from_csv
from LSTM_model.utils import get_trainer
from LSTM_model.config import TRAIN_CKPT_PATH, TRAIN_CSV_DIR
from LSTM_model.predictions import review_predictions
from LSTM_model.dataloader import FacesFeaturesDataModule
from sklearn.model_selection import train_test_split


def train():
    """Train loop.

    Use "SmileAuthenticityPredictor(num_classes=2, num_features=39)" to create new model.

    Use "SmileAuthenticityPredictor.load_from_checkpoint()" to load from checkpoint.

    Config TRAIN_CKPT_PATH in model.model_config to choose model checkpoint path.

    Config TRAIN_CSV_DIR in model.model_config to choose data directory.
    """
    try:
        csv_directory = TRAIN_CSV_DIR
        data = load_data_from_csv(csv_directory)
        train_data, test_data = train_test_split(
            data, test_size=0.2, random_state=987876
        )
        batch_size = 64
        data_module = FacesFeaturesDataModule(
            train_data, test_data, batch_size=batch_size
        )
        data_module.setup()
        # print(train_data[0][0])

        model = SmileAuthenticityPredictor(num_classes=2, num_features=10)
        # model = SmileAuthenticityPredictor.load_from_checkpoint(
        #     TRAIN_CKPT_PATH, num_classes=2, num_features=39
        # )

        trainer = get_trainer()
        trainer.fit(model, datamodule=data_module)
        trainer.test(datamodule=data_module, ckpt_path="best")
        review_predictions(
            test_data, trainer.checkpoint_callback.best_model_path
        )
    except Exception as e:
        print(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    train()
