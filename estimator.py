import cv2 as cv


class Estimator:
    """ This is a sample tracker, replace this class with your own tracker! """
    def __init__(self):
        '''
        TODO
        '''

        self.model = self.model_load()
    
    def preprocess(self,img_path):
        '''
        TODO
        '''
        img = None

        return img
    
    def model_load(self,):
        '''
        TODO
        ''' 

        model = None       

        return model
    
    def predict(self, img_path):

        img = self.preprocess(img_path)

        '''
        TODO
        '''

        pose = self.model(img)

        return pose



