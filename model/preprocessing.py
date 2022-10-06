from this import d
import numpy as np
import cv2
from PIL import ImageEnhance
from PIL import Image

class DataPipeline():


    def __init__(self, image, height, width, padding=None):

        self.imageArray = image
        self.h = height
        self.w = width
        self.letters = []
        self.pad = padding
    
    def preprocessing(self, image, padding=None, flatten=False):

        '''
        parameters:
        image (array): Source image to preprocess.
        padding (int): padding factor, if None, no padding is added.
        flatten (bool): if True, the preprocessed image is flattened. 
        '''
        
        color_mode = 'L' # 'L' - (8-bit pixels, black and white)
        padding_mode = 'constant'
        padding_mode_value = 255
        sharpness_factor = 10
        brightness_factor = 2

        img = Image.fromarray(image)
        img = img.convert(mode=color_mode).resize((self.h, self.w), resample=Image.Resampling.BICUBIC)
        img = ImageEnhance.Sharpness(img).enhance(sharpness_factor)
        img = ImageEnhance.Brightness(img).enhance(brightness_factor)
        img = np.asarray(img)

        # adds a padding to the individual character array
        if padding: 
            img = Image.fromarray( np.pad(np.asarray(img), padding, mode=padding_mode, constant_values=padding_mode_value) )
            img = img.convert(mode=color_mode).resize((self.h, self.w), resample=Image.Resampling.BICUBIC)

        if flatten: return np.asarray(img).flatten()

        return np.asanyarray(img)
    
    def get_letters(self):

        gaussian_kernel_blur = (3,3)
        threshold_range = (250,255)
        sigma_x = 6 #standard deviation in x direction.

        grayImageBlured = cv2.GaussianBlur(self.imageArray, gaussian_kernel_blur , sigma_x)
        _, thresh_image = cv2.threshold(self.imageArray, threshold_range[0], threshold_range[1], cv2.THRESH_TOZERO)
        edges=cv2.Canny(image = grayImageBlured, threshold1=100, threshold2=255) #finds edges in the input image.
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            
            epsilon = 0.001*(cv2.arcLength(contour, True))
            approx = cv2.approxPolyDP(contour, epsilon, True)

            #get the index coordinates values of a single digit from the image array based in the contours found.
            (x_min, x_max) = (contour[:][:,:,1].min(), contour[:][:,:,1].max())
            (y_min, y_max) = (contour[:][:,:,0].min(), contour[:][:,:,0].max())

            # filters the contours by number of angles. 
            if len(contour)>=0 and len(contour) <=88:
                image_shape = (28,28,1)
                digit = thresh_image[x_min:x_max,y_min:y_max]
                #inverts the image color
                digit = (255 - self.preprocessing(digit, padding=self.pad, flatten=False)).reshape(image_shape)
                self.letters.append((digit, y_min))
            if len(contour) >=90:
                image_shape = (28,28,1)
                digit = thresh_image[x_min:x_max,y_min:y_max]
                digit = (255 - self.preprocessing(digit, padding=self.pad, flatten=False)).reshape(image_shape)
                self.letters.append((digit, y_min))
        
        return self.letters