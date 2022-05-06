class EDAConfig:
    idx_2_landmark = dict()
    landmarks = ["Nose", 'REye', 'LEye', 'RM', 'LM', 'REar', 'LEar']

    for i, landmark in enumerate(landmarks):
        idx_2_landmark[i] = landmark

    landmark_2_idx = dict(zip(idx_2_landmark.values(), idx_2_landmark.keys()))