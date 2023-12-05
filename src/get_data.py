import os
import yaml
import pandas as pd
import argparse

# TO-DO:
# Read the parameters
# Process the parameters
# Return the dataframe

def read_params(config_path):
    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)
    return params
def get_data(config_path):
    config = read_params(config_path)
    print(config)
    data_path = config['data_source']['s3_source']
    df = pd.read_csv(data_path, sep=",", encoding="utf-8")
    print(df.head())
    return df

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    default_config_path = os.path.join("config", "params.yaml")
    args.add_argument('--config', type=str, default=default_config_path)
    parsed_args = args.parse_args()
    data = get_data(config_path=parsed_args.config)