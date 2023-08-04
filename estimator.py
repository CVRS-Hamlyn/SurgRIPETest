import cv2 as cv


class Estimator:
    """ This is a sample tracker, replace this class with your own tracker! """
    def __init__(self):
        '''
        TODO
        '''

        self.model = self.model_load()
    
    def image_reader(self,img_path):
        '''
        TODO
        read image and preprocessing
        '''
        img = None

        return img
    
    def model_load(self):
        '''
        TODO
        ''' 

        model = None       

        return model
    
    def predict(self, img_path):

        img = self.image_reader(img_path)

        '''
        TODO
        '''

        pose = self.model(img)

        return pose



