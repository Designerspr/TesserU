import cv2
import pytesseract
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering


def reverse(img, area, kernel_size=(3, 3)):
    '''reverse the area in the img only.
    '''
    shape = img.shape
    value = int(np.max(img))
    mask = np.zeros(shape)
    mask = cv2.drawContours(
        mask,
        [area],
        0,
        value,
        thickness=-1,
    )
    kernel = np.ones(kernel_size, dtype=np.int8)
    # mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = np.array(mask, dtype=np.uint8)
    return mask ^ img


def info2array(info):
    single_list = info.split('\n')
    sym, pos = list(), list()
    for single in single_list:
        args = single.split(' ')
        sym.append(args[0])
        pos.append(np.array(args[1:-1]).astype(np.int))
    return sym, pos


def dataReadin(path):
    # data read in
    pic = cv2.imread(path)
    pic = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)
    pic = np.array(pic)
    return pic


def zeroPadding(pic, zeropadding_param):
    # Zero padding
    pic_shape = (int(pic.shape[0] * (1 + zeropadding_param)),
                 int(pic.shape[1] * (1 + zeropadding_param)))
    ll, uh = int(pic_shape[0] * (zeropadding_param / 2)), int(
        pic_shape[1] * (zeropadding_param / 2))
    rl, dh = ll + pic.shape[0], uh + pic.shape[1]
    full = np.zeros(pic_shape, dtype=np.uint8)
    full[ll:rl, uh:dh] = pic
    return full


def toOnlyBackGround(pic, thershold):
    pic_area = pic.shape[0] * pic.shape[1]
    # backup for printing
    # pic_assigned = cv2.cvtColor(pic.copy(), cv2.COLOR_GRAY2RGB)

    # search all the contours
    _, contours, _ = cv2.findContours(pic, cv2.RETR_LIST,
                                      cv2.CHAIN_APPROX_SIMPLE)
    # print('num=', len(contours))

    for contour in contours:
        M = cv2.moments(contour)
        area = M['m00']
        # check if area is big enough and bg=black
        if area / pic_area > thershold:

            # print(len(contour))
            # print(contour[0][0][0], contour[0][0][1])
            #if pic[contour[0][0][1]][contour[0][0][0]] != 0:
            # pic_assigned = cv2.drawContours(
            #    pic_assigned, [contour], 0, (255, 0, 0), thickness=1)
            pic = reverse(pic, contour, kernel_size=(7, 7))

    return pic


def removeLine(pic, lengthThershold, var_thershold):
    # with long structure, lines can be removed by tophat.
    t_height, t_width = pic.shape
    t_height //= lengthThershold
    t_width //= lengthThershold
    aHighStruct = np.ones((t_height, 1), dtype=np.int8)
    aLongStruct = np.ones((1, t_width), dtype=np.int8)
    pic = cv2.morphologyEx(pic, cv2.MORPH_TOPHAT, aHighStruct)
    pic = cv2.morphologyEx(pic, cv2.MORPH_TOPHAT, aLongStruct)

    # add an filter to remove remain fatel elements
    _, contours, _ = cv2.findContours(pic, cv2.RETR_LIST,
                                      cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x1, y1, w, h = cv2.boundingRect(contour)
        var = w / h
        if var < 1:
            var = 1 / var
        if var > 3:
            pic = reverse(pic,contour,)


    return pic


def scale(pic, scale_param):
    # scale the img
    pic_shape = np.array(pic.shape) * scale_param
    pic_shape = (pic_shape[1], pic_shape[0])
    pic = cv2.resize(pic, pic_shape, cv2.INTER_NEAREST)
    # scale the display one
    # pic_assigned = cv2.resize(pic_assigned, pic_shape, cv2.INTER_NEAREST)
    return pic


def display(pic, text, save_or_show='show', outputName='UNAMED'):
    plt.figure()
    plt.subplot(211)
    plt.axis('off')
    plt.imshow(pic, cmap='binary_r')
    plt.subplot(212)
    plt.axis('off')
    plt.text(0, 0, text, fontsize=10)
    # plt.imshow(pic_assigned)
    if save_or_show == 'show':
        plt.show()
    elif save_or_show == 'save':
        outputName = 'result_2\\' + outputName
        plt.savefig(outputName)
    plt.close('all')


def toString(path,
             model_used,
             zero_param=0.1,
             thershold=0.005,
             lengthThershold=9,
             var_thershold=3,
             scale_param=3):
    pic = dataReadin(path)
    nameOnly=path.split('.')[0]
    pic = zeroPadding(
        pic,
        zeropadding_param=zero_param,
    )
    # flip it
    pic = cv2.flip(pic, 0)

    # checkpoint output 1
    # text = pytesseract.image_to_string(pic, lang=model_used)
    # display(pic, text, save_or_show='show',outputName=nameOnly+'_1.png')

    pic = toOnlyBackGround(pic, thershold)

    # checkpoint output 2
    # text = pytesseract.image_to_string(pic, lang=model_used)
    # display(pic, text, save_or_show='show',outputName=nameOnly+'_2.png')

    pic = removeLine(
        pic, lengthThershold=lengthThershold, var_thershold=var_thershold)

    # checkpoint output 3
    # text = pytesseract.image_to_string(pic, lang=model_used)
    # display(pic, text, save_or_show='show',outputName=nameOnly+'_3.png')

    pic = scale(pic, scale_param=scale_param)

    # main detection
    '''
    info = pytesseract.image_to_boxes(pic, lang='eng')
    height = pic_shape[1]
    symbol, corners = info2array(info)
    size,pos=list(),list()
    for sym, cor in zip(symbol, corners):
        cor1, cor2 = (cor[0], height - cor[1]), (cor[2], height - cor[3])
        # pic_assigned = cv2.rectangle(pic_assigned, cor1, cor2, (0, 255, 0), 2)
    '''
    # checkpoint output 4
    text = pytesseract.image_to_string(pic, lang=model_used)
    display(pic, text, save_or_show='show',outputName=nameOnly+'_4.png')


def main():
    nameList = os.listdir()
    for name in nameList:
        if name.endswith('2.bmp'):
            print('processing with pic',name)
            toString(name, model_used='judge')

main()