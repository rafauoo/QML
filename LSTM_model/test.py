from LSTM_model.lstm import SmileAuthenticityPredictor
from LSTM_model.data import load_data_from_csv
from LSTM_model.utils import get_trainer
import os
from LSTM_model.predictions import review_predictions, review_predictions2
from LSTM_model.dataloader import FacesFeaturesDataModule

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.abspath(os.path.join(os.sep, ROOT_DIR, "out"))
def test(num):
    """Test model on whole data.

    Config TEST_CKPT_PATH in model.model_config to choose model checkpoint path.

    Config TEST_CSV_DIR in model.model_config to choose data directory.
    """
    CKPT_PATH = os.path.abspath(os.path.join(os.sep, ROOT_DIR, "experiments", str(num), "checkpoint.ckpt"))
    try:
        csv_directory = TEST_DIR
        data = load_data_from_csv(csv_directory)
        train_data = data
        test_data = data
        batch_size = 64
        data_module = FacesFeaturesDataModule(
            train_data, test_data, batch_size=batch_size
        )
        data_module.setup()
        trainer = get_trainer()
        model = SmileAuthenticityPredictor(num_classes=2, num_features=10)
        trainer.test(
            model=model,
            datamodule=data_module,
            ckpt_path=CKPT_PATH,
        )
        review_predictions(
            test_data=test_data,
            num=num,
            ckpt_path=CKPT_PATH,
        )
    except Exception as e:
        print(f"Error during training: {e}")
        raise

def test2(num):
    """Test model on whole data.

    Config TEST_CKPT_PATH in model.model_config to choose model checkpoint path.

    Config TEST_CSV_DIR in model.model_config to choose data directory.
    """
    CKPT_PATH = os.path.abspath(os.path.join(os.sep, ROOT_DIR, "experiments", str(num), "checkpoint.ckpt"))
    try:
        csv_directory = TEST_DIR
        data = load_data_from_csv(csv_directory)
        train_data = data
        test_data = data
        batch_size = 64
        data_module = FacesFeaturesDataModule(
            train_data, test_data, batch_size=batch_size
        )
        data_module.setup()
        trainer = get_trainer()
        model = SmileAuthenticityPredictor(num_classes=2, num_features=10)
        trainer.test(
            model=model,
            datamodule=data_module,
            ckpt_path=CKPT_PATH,
        )
        return review_predictions2(
            test_data=test_data,
            num=num,
            ckpt_path=CKPT_PATH,
        )
    except Exception as e:
        print(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    test()
