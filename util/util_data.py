import pandas as pd


def load_path(root_path):
    paths = pd.read_csv(f"{root_path}/data/info.csv", index_col=0)
    paths.columns = ["Human_id", "Path"]
    return paths
