import cv2 as cv
import numpy as np
import math



IMAGE_PATH = "./Images/Lena.png"
OUTPUT_PATH = "./Images/result.jpg"
RADIUS = 5
SIGMA = 1
TL_PARA = 0.1
TH_PA = 0.5


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
        self.dx = np.zeros([h - 1,w - 1])
        self.dy = np.zeros([h - 1,w - 1])
        self.d = np.zeros([h - 1,w - 1])
        for i in range(h - 1):
            for j in range(w - 1):
                self.dx[i][j] = trans[i,j + 1] - trans[i,j]
                self.dy[i][j] = trans[i + 1,j] - trans[i,j]
                self.d[i][j] = np.sqrt(np.square(self.dx[i,j]) + np.square(self.dy[i,j]))
                # print(i,j)


    def myNMS(self):
        self.NMS = np.copy(self.d)
        h,w = self.NMS.shape
        self.NMS[0,:] = self.NMS[h - 1,:] = self.NMS[:,w - 1] = self.NMS[:,0] = 0
        for i in range(1,h - 1):
            for j in range(1,w - 1):
                if self.d[i,j] == 0:
                    self.NMS[i,j] = 0
                else:
                    grad_x = self.dx[i,j]
                    grad_y = self.dy[i,j]
                    grad_temp = self.d[i,j]
                    if np.abs(grad_x) < np.abs(grad_y):
                        weight = np.abs(grad_x) / np.abs(grad_y)
                        grad2 = self.d[i - 1,j]
                        grad4 = self.d[i + 1,j]
                        if grad_x * grad_y > 0:
                            grad1 = self.d[i - 1,j - 1]
                            grad3 = self.d[i + 1,j + 1]

                        else:
                            grad1 = self.d[i - 1,j + 1]
                            grad3 = self.d[i + 1,j - 1]

                    else:
                        weight = np.abs(grad_y) / np.abs(grad_x)
                        grad2 = self.d[i, j - 1]
                        grad4 = self.d[i, j + 1]
                        if grad_x * grad_y > 0:
                            grad1 = self.d[i + 1, j - 1]
                            grad3 = self.d[i - 1, j + 1]

                        else:
                            grad1 = self.d[i - 1, j - 1]
                            grad3 = self.d[i + 1, j + 1]

                    gradTemp1 = weight * grad1 + (1 - weight) * grad2
                    gradTemp2 = weight * grad3 + (1 - weight) * grad4
                    if grad_temp >= gradTemp1 and grad_temp >= gradTemp2:
                        self.NMS[i, j] = grad_temp
                    else:
                        self.NMS[i, j] = 0



    def canny(self):
        self.myGaussian(self.img)
        self.getGradient()
        self.myNMS()
        h,w = self.NMS.shape
        result = np.zeros([h,w])
        TL = TL_PARA * np.max(self.NMS)
        TH = TH_PA * np.max(self.NMS)
        for i in range(h):
            for j in range(w):
                if (self.NMS[i, j] < TL):
                    result[i, j] = 0

                elif (self.NMS[i, j] > TH):
                    result[i, j] = 255

                # elif i == 1 and j == 1 and self.myBFS(i,j,self.NMS,TL,TH):
                #     result[i][j] = 255
                # print((self.NMS[i, [j-1, j+1]] < TH).any())
                elif (self.NMS[i - 1, j - 1:j + 1] < TH).any() or (self.NMS[i + 1, j - 1:j + 1] < TH).any() or (self.NMS[i - 1 : i + 1, j - 1] < TH).any()\
                        or (self.NMS[i + 1 : i + 1, j - 1] < TH).any():
                    result[i, j] = 255
        cv.imwrite(OUTPUT_PATH,result)


    # def myBFS(self,x,y,matrix,TL,TH):
    #     if x < 0 or y < 0 or x >= len(matrix) or y >= len(matrix[0]):
    #         return False
    #     if matrix[x][y] > TH:
    #         return True
    #     if matrix[x][y] < TL:
    #         return False
    #
    #     return self.myBFS(x + 1,y,matrix,TL,TH) or self.myBFS(x - 1,y,matrix,TL,TH) or self.myBFS(x,y - 1,matrix,TL,TH) or self.myBFS(x,y + 1,matrix,TL,TH)




if __name__ == '__main__':
    t = Main(IMAGE_PATH)
    t.canny()

