import cv2
import os


def flip_img_and_save(img, file_name):
    vertical_flip = cv2.flip(img, 0)
    horizontal_flip = cv2.flip(img, 1)
    both_flip = cv2.flip(img, -1)

    cv2.imwrite(file_name + '_v' + '.jpg', vertical_flip)
    cv2.imwrite(file_name + '_h' + '.jpg', horizontal_flip)
    cv2.imwrite(file_name + '_b' + '.jpg', both_flip)


def run(folder):
    path = './datasets/' + folder + '/'
    img_names = os.listdir(path)
    for img_name in img_names:
        img = cv2.imread(path + img_name)
        flip_img_and_save(img, path+img_name.split('.')[0])


if __name__ == '__main__':
    run(input('Sub-folder: '))