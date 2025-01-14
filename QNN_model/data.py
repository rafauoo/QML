import os
import pandas as pd


def load_data_from_csv(csv_dir: str) -> list[pd.DataFrame]:
    """Loads data from given directory.

    :param csv_dir: data directory
    :type csv_dir: str
    :return: list of DataFrames containing data from the directory
    :rtype: list[pd.DataFrame]
    """
    data = []
    for file_name in os.listdir(csv_dir):
        file_name: str
        if file_name.endswith(".csv"):
            file_path = os.path.join(csv_dir, file_name)
            df = pd.read_csv(file_path)
            line_count = len(df)
            if line_count == 39:
                angles = df["lips_corners_from_nose_angle"].tolist()[:10]
                is_authentic = 1 if "spontaneous" in file_name else 0
                data.append((angles, is_authentic))
            else:
                print(
                    f"Skipped {file_name} due to incorrect number of lines: {line_count}"
                )
    return data
