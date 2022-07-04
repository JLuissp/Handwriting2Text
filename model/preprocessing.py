import numpy as np
import cv2
from PIL import ImageEnhance
from PIL import Image

class DataPipeline():
    def __init__(self, image, h, w, padding=None):
        self.imageArray = image
        self.h = h
        self.w = w
        self.letters = []
        self.pad = padding
    
    def preprocessing(self, image, padding=None, flatten=False):
        img = Image.fromarray(image)
        img = img.convert('L').resize((self.h, self.w), resample=Image.Resampling.BICUBIC)
        img = ImageEnhance.Sharpness(img).enhance (10)
        img = ImageEnhance.Brightness(img).enhance(2)
        img = np.asarray(img)
        if padding: 
            img = Image.fromarray(np.pad(np.asarray(img), padding, mode='constant', constant_values=255))
            img = img.convert('L').resize((self.h, self.w), resample=Image.Resampling.BICUBIC)
        if flatten: return np.asarray(img).flatten()

        return np.asanyarray(img)
    
    def get_letters(self):
        
        grayImageBlured = cv2.GaussianBlur(self.imageArray, (3,3), 6)
        _, thresh_image = cv2.threshold(self.imageArray, 250, 255, cv2.THRESH_TOZERO)
        edges=cv2.Canny(image = grayImageBlured, threshold1=250, threshold2=255)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
        
            epsilon = 0.001*(cv2.arcLength(contour, True))
            approx = cv2.approxPolyDP(contour, epsilon, False)

            (x_min, x_max) = (contour[:][:,:,1].min(), contour[:][:,:,1].max())
            (y_min, y_max) = (contour[:][:,:,0].min(), contour[:][:,:,0].max())


            if len(contour)>=0 and len(contour) <=50:
                digit = thresh_image[x_min:x_max,y_min:y_max]
                self.letters.append((255 - self.preprocessing(digit, padding=self.pad, flatten=False)).reshape(28,28,1))
            if len(contour) >=90:
                digit = thresh_image[x_min:x_max,y_min:y_max]
                self.letters.append((255 - self.preprocessing(digit, padding=self.pad, flatten=False)).reshape(28,28,1))
        
        return self.letters