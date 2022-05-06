import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import combinations
import matplotlib.pyplot as plt

from util.util_data import load_path
from util.util_eda import load_landmark, pairwise_distance
from config import EDAConfig


def main():
    # # Save landmark
    # info = load_path(root_path)
    # save_landmark(info)

    # # Save Pairwise Distance
    # save_pair_dis()

    pairs = tuple(combinations((eda_config.landmark_2_idx.keys()), 2))
    for pair in pairs:
        compare(pair)
    print()


def save_landmark(info):
    human_landmark = pd.DataFrame()
    for human in tqdm(np.unique(info['Human_id'])):
        landmark = load_landmark(root_path, info.loc[info['Human_id'] == human, "Path"], eda_config.idx_2_landmark)
        landmark["Human_id"] = human

        human_landmark = pd.concat((human_landmark, landmark), axis=0)
    human_landmark.to_csv(f"{root_path}/save/landmarks.csv")


def save_pair_dis():
    human_landmark = pd.read_csv(f"{root_path}/save/landmarks.csv", index_col=0)
    human_pair_dis = pd.DataFrame()

    for human in tqdm(np.unique(human_landmark['Human_id'])):
        human_data = human_landmark.loc[human_landmark["Human_id"] == human]

        for photo in np.unique(human_data['Photo_id']):
            human_photo = human_data.loc[human_data["Photo_id"] == photo]
            dis = pairwise_distance(human_photo[['X', 'Y']].values)

            for i, j in list(combinations(human_photo['Part'], 2)):
                pair_dis = dis[eda_config.landmark_2_idx[i], eda_config.landmark_2_idx[j]]
                human_pair_dis = pd.concat((human_pair_dis, pd.DataFrame([[human, (i, j), pair_dis]],
                                                                         columns=["Human_id", 'Pair', 'Distance'])),
                                           axis=0)
    human_pair_dis.to_csv(f"{root_path}/save/pair_dis.csv")


def compare(pair: tuple):
    pairwise_dis = pd.read_csv(f"{root_path}/save/pair_dis.csv", index_col=0)
    target = pairwise_dis.groupby(['Human_id', 'Pair'], as_index=False).mean()
    target = target[target['Pair'] == str(pair)]
    if len(target.values) == 0:
        target = target[target['Pair'] == str((pair[1], pair[0]))]

    plt.figure()
    plt.hist(target['Distance'].to_list())
    plt.title(f"Case : {str(pair)}")
    plt.xlabel("Distance", fontsize=15)
    plt.ylabel("Count", fontsize=15)
    plt.savefig(f"{root_path}/save/eda/{str(pair)}.png")
    plt.close()


if __name__ == "__main__":
    print("Hello, world!")
    root_path = os.getcwd()
    eda_config = EDAConfig()
    main()
