import cv2
import numpy as np
from skimage import transform as trans

src1 = np.array([[51.642, 50.115], [57.617, 49.990], [35.740, 69.007],
                 [51.157, 89.050], [57.025, 89.702]],
                dtype=np.float32)
#<--left
src2 = np.array([[45.031, 50.118], [65.568, 50.872], [39.677, 68.111],
                 [45.177, 86.190], [64.246, 86.758]],
                dtype=np.float32)

#---frontal
src3 = np.array([[39.730, 51.138], [72.270, 51.138], [56.000, 68.493],
                 [42.463, 87.010], [69.537, 87.010]],
                dtype=np.float32)

#-->right
src4 = np.array([[46.845, 50.872], [67.382, 50.118], [72.737, 68.111],
                 [48.167, 86.758], [67.236, 86.190]],
                dtype=np.float32)

#-->right profile
src5 = np.array([[54.796, 49.990], [60.771, 50.115], [76.673, 69.007],
                 [55.388, 89.702], [61.257, 89.050]],
                dtype=np.float32)

src = np.array([src1, src2, src3, src4, src5]) # 
src_map = {112: src, 224: src * 2}

arcface_src = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

arcface_src *= 1.3
arcface_src -= np.array([20, 0])
arcface_src -= np.array([0, 35])



arcface_src = np.expand_dims(arcface_src, axis=0)

# In[66]:


# lmk is prediction; src is template
# M: Homogeneous transformation matrix
def estimate_norm(lmk, image_size=112, mode='arcface'):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    # 1를 axis = 1 방향으로 index 2자리에 삽입
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')
    if mode == 'arcface':
        if image_size == 112:
            src = arcface_src
        else:
            src = float(image_size) / 112 * arcface_src
    else:
        src = src_map[image_size]
    for i in np.arange(src.shape[0]):
        # estimate: transform 할 때 각도를 지정하지 않고, 원래 좌표와 목적지 좌표를 지정해서 transform함.
        # transform할 때 다섯 개의 좌표의 평균을 구하는 과정이 있기 때문에, 모든 점의 좌표가 의미가 있음.
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i])**2, axis=1)))
        #         print(error)
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index


def norm_crop(img, route, landmark, image_size=112, mode='arcface'):
    M, pose_index = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    # normalize
    warped = cv2.normalize(warped, None, 0, 255, cv2.NORM_MINMAX)
    # equalizeHist
    # warped = cv2.equalizeHist(warped)
    # clahe
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # warped = clahe.apply(warped)
    
    # route 영역만 남기고 zero padding 적용(헤어스타일, 하관 제거)
    route_tran = np.insert(route, 2, values=np.ones(9), axis=1)
    warped_route = np.dot(M, route_tran.T) 
    warped_route = warped_route.T     # 정면 방향에서의 route 좌표
    warped_route = np.asarray(warped_route, dtype = int)
    # print(warped_route)
    # print(type(warped_route))
    # 남은 부분 zero padding 적용
    mask = np.zeros((warped.shape[0], warped.shape[1])) 
    mask = cv2.fillConvexPoly(mask, warped_route, 1) 
    mask = mask.astype(np.bool) 
    
    out = np.zeros_like(warped) 
    out[mask] = warped[mask]
    return out