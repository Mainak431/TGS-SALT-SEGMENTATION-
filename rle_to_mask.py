import numpy as np  # linear algebra
import csv
import cv2

def rleToMask(path,name,rleNumbers, height, width):
    rows, cols = height, width
    #rleNumbers = [int(numstring) for numstring in rleString.split(' ')]
    rlePairs = np.array(rleNumbers).reshape(-1, 2)
    img = np.zeros(rows * cols, dtype=np.uint8)
    for index, length in rlePairs:
        index -= 1
        img[index:index + length] = 255
    img = img.reshape(cols, rows)
    img = img.T
    seg_img = cv2.resize(img, (101, 101))
    cv2.imwrite(path + name + '.jpg', seg_img)
    return img

path1 = '/home/titanx/Desktop/Mainak/TGS SALT/train.csv'

path = '/home/titanx/Desktop/Mainak/TGS SALT/create_mask/'
listi = []
with open(path1,'r') as f:
    reader = csv.reader(f,delimiter=',')
    listi = list(reader)[1:]

for i in listi:
    print(i[0])
    print(i[1])
    li = []
    l = i[1].split()
    for j in l:
        li.append(int(j))
    print(li)
    r = rleToMask(path,i[0],li,101,101)