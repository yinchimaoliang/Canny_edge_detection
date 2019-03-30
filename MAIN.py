import cv2 as cv
import numpy as np



IMAGE_PATH = "./Images/Dorothea.jpg"


class Main():
    def __init__(self,path):
        self.img = cv.imread(path,0)



        