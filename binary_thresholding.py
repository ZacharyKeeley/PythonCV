import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

if __name__ == "__main__":

    img = cv2.imread("/home/zack/Desktop/PythonCV/images/image06.jpg", 0)
    ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
    ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
    ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
    ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)

    titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO', 'TOZERO_INV']
    images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

    for i in xrange(6):
        plt.subplot(2,3, i+1),plt.imshow(images[i],)
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])

    plt.show()

    '''
    # reading in the file
    img = cv2.imread("/home/zack/Desktop/PythonCV/images/image05.jpg", 0)
    
    # converting it to black and white
    ret, thresh1 = cv2.threshold(img, 200,200, cv2.THRESH_BINARY)

    # showing the image
    cv2.imshow(img)
    '''