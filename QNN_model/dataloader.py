import torch
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader


class FacesFeaturesDataset(Dataset):
    """Faces features dataset handler."""

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def __len__(self) -> pd.DataFrame:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[torch.Tensor, torch.Tensor]:
        ffs, auth = self.data[idx]
        return dict(
            faces_features=torch.Tensor(ffs),
            authenticity=torch.tensor(auth).long(),
        )


class FacesFeaturesDataModule(pl.LightningDataModule):
    """Faces features data module handler."""

    def __init__(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        batch_size: int,
    ):
        super().__init__()
        self.train_data = train_data
        self.test_data = test_data
        self.test_dataset = None
        self.train_dataset = None
        self.batch_size = batch_size

    def setup(self, stage: str = None):
        self.train_dataset = FacesFeaturesDataset(self.train_data)
        self.test_dataset = FacesFeaturesDataset(self.test_data)

    def train_dataloader(self) -> DataLoader:
        """Creates train DataLoader.

        :return: train DataLoader
        :rtype: DataLoader
        """
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        """Creates validation DataLoader.

        :return: validation DataLoader
        :rtype: DataLoader
        """
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False
        )

    def test_dataloader(self) -> DataLoader:
        """Creates test DataLoader.

        :return: test DataLoader
        :rtype: DataLoader
        """
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False
        )
