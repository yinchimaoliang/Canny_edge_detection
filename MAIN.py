import cv2 as cv
import numpy as np
import math



IMAGE_PATH = "./Images/Dorothea.jpg"
OUTPUT_PATH = "./Images/result.jpg"
RADIUS = 5
SIGMA = 5


class Main():
    def __init__(self,path):
        self.img = cv.imread(path,0)



    def getCv(self,r, sigma):
        return 1 / (2 * math.pi * sigma ** 2) * math.exp((-r ** 2) / (2 * sigma ** 2))




    def getFilter(self,radius,sigma):
        window = np.zeros((radius * 2 + 1, radius * 2 + 1))
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                r = (i ** 2 + j ** 2) ** 0.5
                window[i + radius][j + radius] = self.getCv(r, sigma)
        return window / np.sum(window)

    def myGaussian(self, src):
        myFilter = self.getFilter(RADIUS, SIGMA)
        trans = cv.filter2D(src, 1, myFilter)

        cv.imwrite(OUTPUT_PATH,trans)
        return trans



    def getGradient(self):
        trans = self.myGaussian(self.img)
        h,w = trans.shape
        dx = np.zeros([h - 1,w - 1])
        dy = np.zeros([h - 1,w - 1])
        self.d = np.zeros([h - 1,w - 1])
        for i in range(h - 1):
            for j in range(w - 1):
                dx[i][j] = trans[i,j + 1] - trans[i,j]
                dy[i][j] = trans[i + 1,j] - trans[i,j]
                self.d[i][j] = np.sqrt(np.square(dx[i,j]) + np.square(dy[i,j]))
                # print(i,j)


    def canny(self):
        # self.myGaussian(self.img)
        self.getGradient()






if __name__ == '__main__':
    t = Main(IMAGE_PATH)
    t.canny()

