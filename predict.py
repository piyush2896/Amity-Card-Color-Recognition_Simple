import numpy as np
import cv2
from classifier import KNN
import matplotlib.pyplot as plt
import os


def get_data(sub_folders):
    path = './datasets/'
    imgs = []
    labels = []
    for folder in sub_folders:
        cur_folder = path + folder
        img_list = os.listdir(cur_folder)
        for img_name in img_list:
            img = cv2.imread(cur_folder + '/' + img_name)
            img = cv2.resize(img, (400, 250))
            img = img.reshape(400 * 250 * 3,)
            imgs.append(img)
            labels.append(folder)
    imgs = np.asarray(imgs)
    labels = np.asarray(labels)
    return imgs, labels


def get_classifier(X_train, Y_train, k=3):
    knn = KNN(5)
    knn.fit(X_train, Y_train)
    return knn


def get_data_():
    # get data
    X, Y = get_data(['Orange', 'Green', 'White'])
    Y_ = np.zeros(Y.shape)
    Y_[Y == 'Green'] = 1
    Y_[Y == 'White'] = 2
    Y_ = Y_.astype('float32')

    # shuffle data
    all_data = np.zeros((X.shape[0], X.shape[1] + 1))
    all_data[:, :-1] = X
    all_data[:, -1] = Y_
    np.random.shuffle(all_data)

    # split into X and Y
    X, Y_ = all_data[:, :-1], all_data[:, -1]

    return X, Y_


def run(img):
    # get data
    X, Y_ = get_data_()

    # get classifier
    knn = get_classifier(X, Y_, k=5)

    # get image
    img = cv2.resize(img, (400, 250))
    img = img.reshape(400 * 250 * 3,)

    # predict
    pred = knn.predict(np.array([img]))
    if pred == 0: return 'Orange'
    elif pred == 1: return 'Green'
    return 'White'


def check_run():

    # split into train and test
    X, Y_ = get_data_()
    split = int(0.9 * X.shape[0])
    X_train, X_test = X[:split, :], X[split:, :]
    Y_train, Y_test = Y_[:split], Y_[split:]
    
    # make a classifier
    knn = get_classifier(X_train, Y_train)
    return knn.accuracy(X_test, Y_test)


if __name__ == '__main__':
    print(check_run())
    file = input('Enter File location: ')
    img = cv2.imread(file)
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img1)
    plt.show()
    print(run(img))
