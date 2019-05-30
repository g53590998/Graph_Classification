import cv2
import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd
#from pandas import Series, DataFrame
img = cv2.imread('101_ObjectCategories/test_graph/069_0066.jpg')
#img = cv2.resize(img, (100, 52), interpolation=cv2.INTER_CUBIC)
#plt.imshow(img)
#plt.show()
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,binImg = cv2.threshold(grayImg, 100, 255, cv2.THRESH_BINARY)
contours = cv2.findContours(binImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,hierarchy=None)
cv2.drawContours(img, contours[1], -1, (255, 255, 255), 2)
plt.imshow(img)
plt.show()
#hierarchyDF = DataFrame(hierarchy[0], columns = ['pre', 'next', 'child', 'parent'])

