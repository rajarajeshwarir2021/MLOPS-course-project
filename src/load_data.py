# TO-DO:
# Read the data from the datasource
# Save it to the data/raw folder for further processing

import os
from get_data import read_params, get_data
import argparse


def load_and_save(config_path):
    """
    Read the data from the datasource and save it to the data/raw folder for further processing
    """
    config = read_params(config_path)
    df = get_data(config_path)
    new_cols = [col.replace(" ", "_") for col in df.columns]
    raw_data_path = config["load_data"]["raw_dataset_csv"]
    df.to_csv(raw_data_path, sep=",", encoding="utf-8", index=False, header=new_cols)
    print(new_cols)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    default_config_path = os.path.join("config", "params.yaml")
    args.add_argument('--config', type=str, default=default_config_path)
    parsed_args = args.parse_args()
    load_and_save(config_path=parsed_args.config)
