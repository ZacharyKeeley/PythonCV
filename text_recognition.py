import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

if __name__ == "__main__":
    file_path_2 = "/home/zack/Desktop/PythonCV/images/image03.jpg"

    '''
    file_path = "/home/zack/Desktop/PythonCV/images/image05.jpg"
    im = Image.open(file_path_2)
    im = im.filter(ImageFilter.MedianFilter())
    enhancer = ImageEnhance.Contrast(im)
    im = enhancer.enhance(2)
    im = im.convert('1')
    '''

    text = pytesseract.image_to_string(Image.open(file_path_2))
    if text:
        print(text)


'''
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import cv2 as cv
import numpy

if __name__ == "__main__":
    video_capture = cv.VideoCapture(0)

    while True:
        ret_val, frame = video_capture.read()
        # frame = frame.filter(ImageFIlter.MedianFilter())
        # enhancer = ImageEnhance.Contrast(frame)
        # frame = enhancer.enhance(2)
        # frame = frame.convert('1')

        text = pytesseract.image_to_string(Image.open(frame))

        if text != null:
            print(text)
        
        cv.imshow(frame)

        if cv.waitKey(10) == 27:
            break

    video_capture.release()
    cv.destroyAllWindows()        
'''