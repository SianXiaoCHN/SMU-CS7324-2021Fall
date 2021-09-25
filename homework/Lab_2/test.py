
from skimage.feature import match_descriptors, daisy
from multiprocessing import Pool
import random
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from time import time


def read_image(name):
    path = 'data/image/' + name
    img = cv2.imread(path, 0)
    img_re = cv2.bitwise_not(img)
    img_new = img_re/255.0
    return img_new


def create_image_file_name(x):
    # input_{suite_id}_{sample_id}_{code}.jpg
    file_name = f"input_{x[0]}_{x[1]}_{x[2]}.jpg"
    return file_name


def train_test_split(X, y):
    X_train, X_test, y_train, y_test = [], [], [], []
    for code in range(15):
        index = list(range(1000))
        random.seed(7324)
        random.shuffle(index)
        for i in index[:500]:
            X_train.append(X[code*1000+i])
            y_train.append(y[code*1000+i])
        for i in index[500:]:
            X_test.append(X[code*1000+i])
            y_test.append(y[code*1000+i])

    return X_train, X_test, y_train, y_test


def extract_descriptors(imgSet):
    desSet = []
    for img_1d in imgSet:
        img_2d = img_1d.reshape(64, 64)
        fea = daisy(img_2d, step=5, radius=15, rings=2,
                    histograms=8, orientations=8, visualize=False)
        des = fea.reshape(fea.shape[0]*fea.shape[1], -1)
        desSet.append(des)

    return desSet


def massive_matching_DAISY(des_img):
    match_numbers = []
    for i, des in enumerate(X_train):
        matches = match_descriptors(
            des_img, des, cross_check=True,  metric=None, p=2, max_distance=np.inf, max_ratio=1.0)
        match_numbers.append((matches.shape[0], i))

    match_numbers.sort(key=lambda x: -x[0])  # sort as most matching points
    return y_train[match_numbers[0][1]], y_train[match_numbers[1][1]], y_train[match_numbers[2][1]]


now = time()

data = pd.read_csv('data/chinese_mnist.csv', encoding='utf-8')
data["image_file"] = data.apply(create_image_file_name, axis=1)

X = np.stack(data['image_file'].apply(read_image)).reshape(-1, 64*64)
y = data['code'].to_numpy()


X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train = extract_descriptors(X_train)
X_test = extract_descriptors(X_test)


with Pool(36) as p:
    results = p.map(massive_matching_DAISY, [i for i in X_test])


top1, top2, top3 = [0]*15, [0]*15, [0]*15
for i, des in enumerate(X_test):
    one, two, three = results[i]
    print(f"{i} / {len(X_test)}")
    if one == y_test[i]:
        top1[y_test[i]] += 1
    elif two == y_test[i]:
        top2[y_test[i]] += 1
    elif three == y_test[i]:
        top3[y_test[i]] += 1
fail = [500-top1[i]-top2[i]-top3[i] for i in range(15)]


print(f"succedd as top1 for {round(sum(top1)/len(X_test)*100,2)}%")
print(f"succedd as top2 for {round(sum(top2)/len(X_test)*100,2)}%")
print(f"succedd as top3 for {round(sum(top3)/len(X_test)*100,2)}%")
print(f"can't be found as top3 for {round(sum(fail)/len(X_test)*100,2)}%")

# plt.plot()
# add plotting of top123 against code, bar, stack

print(f"It took {time()-now} seconds.")
