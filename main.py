import cv2
import numpy as np
import myutils
from imutils import contours

#Credit Card And Template image data
image_path="images/credit_card_05.png"
template_path="images/ocr_a_reference.png"
#Read...
card_img=cv2.imread(image_path)
template_img=cv2.imread(template_path)
myutils.cv_show("img",card_img)
myutils.cv_show("template",template_img)

#Process the template img
digits=myutils.template_img_process(template_img)


#Process the card img
myutils.card_img_process(card_img,digits)





