import numpy as np
import pandas as pd


def read_txt(path, idx_2_landmark: dict):
    result = []
    count = 0
    with open(path) as f:
        while count < len(idx_2_landmark.keys()):
            line = f.readline()
            count += 1
            line = line.replace("\n", "", 1)
            line = line.split("\t")
            result.append(line)
    result = np.array(result, dtype=int)
    return result


def load_landmark(root_path, paths, idx_2_landmark):
    landmarks = pd.DataFrame()
    photo_id = 0
    for path in paths:
        try:
            landmark = read_txt(f"{root_path}/data/High_Resolution_compressed/{path}.txt", idx_2_landmark)
            landmark = pd.DataFrame(landmark)
            landmark.columns = ['X', 'Y']

            landmark['Part'] = list(idx_2_landmark.values())
            landmark["Photo_id"] = photo_id
            photo_id += 1

            landmarks = pd.concat((landmarks, landmark), axis=0)
        except FileNotFoundError:
            continue
    return landmarks


def pairwise_distance(data):
    x = data[:, 0][:, np.newaxis]
    y = data[:, 1][:, np.newaxis]

    result = - 2* (data@data.T)

    result += x ** 2
    result += x.T ** 2
    result += y ** 2
    result += y.T ** 2
    return np.sqrt(result)
