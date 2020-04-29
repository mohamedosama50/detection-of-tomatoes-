######image88
import cv2
import  numpy as np
from  matplotlib import pyplot as plt
img=cv2.imread("33.jpg")# Reading the image
image=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img=cv2.resize(img,(512,512))   #Resizing the image
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV) #convert BGR to HSV

#########################################################

lower_red= np.array([0,50,50])  #hsv for lower range of red
upper_red = np.array([10,255,255]) #hsv for upper range of red
mask = cv2.inRange(hsv, lower_red, upper_red) # making mask
##################################################

lower_green= np.array([36,25,25])  #hsv for lower range of red
upper_green = np.array([80,255,255]) #hsv for upper range of red
mask1 = cv2.inRange(hsv, lower_green, upper_green) # making mask

##################################################
res_r = cv2.bitwise_and(img,img, mask= mask) # ِANDing the mask and the image to show the red only
res_g=cv2.bitwise_and(img,img, mask= mask1) # ANDِing the mask and the image to show the green only
tot=cv2.bitwise_or(res_r,res_g) # ORing res_r with res_g


#############make our full image equal to the result of  ِANDing the mask , identify the face of tomato#######################
full =res_r
full =cv2.cvtColor(full,cv2.COLOR_BGR2RGB)
face=cv2.imread('22.jpg')
face=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)

###################make another variable equal to the original full image to draw the rectangles in it##########################
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img2 =cv2.resize(img2,(512,512))
#######################loop for detection our tomatoes#######################

i=0
while 1 :

 res =cv2.matchTemplate(full,face,cv2.TM_SQDIFF)  ##for matching process between the face of tomato and full image
 min_val,max_val,min_loc,max_loc =cv2.minMaxLoc(res)###knowing the coordinates of result image in matching process
 height,width,channel=face.shape   ###knowing the coordinates of face image
 top_left =min_loc                 ##setting the top left of res equal to min_loc of the heat map image
 bottom_right=(top_left[0]+width,top_left[1]+height)##setting the coordinate of bottom_right  of the heat map image
 cv2.rectangle(full, top_left, bottom_right, (0, 0, 255), 5)
 cv2.rectangle(img2,top_left,bottom_right,(0,0,255),10)
 i=i+1
 if i==14 :
     break


#######showing the results using matplotlib method###############

titles =['original image','mask detection','full detection']
images=[image,full,img2]
for i in range(3):
    plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])




plt.show()