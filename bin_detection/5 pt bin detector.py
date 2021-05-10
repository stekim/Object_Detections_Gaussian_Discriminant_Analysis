'''
ECE276A WI21 PR1: Color Classification and Recycling Bin Detection
'''
import sys
sys.path.append("../")
import numpy as np
import cv2
# from skimage.measure import label, regionprops
from bin_detection.pixel_classifier_bins import PixelClassifier
import matplotlib.pyplot as plt
# import skimage
# from skimage.measure import label, regionprops


class BinDetector():
	def __init__(self):
		'''
			Initilize your stop sign detector with the attributes you need,
			e.g., parameters of your classifier
		'''
		pass

	def segment_image(self, img):
		'''
			Obtain a segmented image using a color classifier,
			e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture, 
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				mask_img - a binary image with 1 if the pixel in the original image is blue and 0 otherwise
		'''
		h,w,d = img.shape[0],img.shape[1],img.shape[2]
		img = img.reshape(h*w, d)
		# cvt from BGR2RGB
		clf = PixelClassifier() 
		y = clf.classify(img)
		y = y.reshape(h,w)
		y = y.astype(np.uint8) ##converted
# 		plt.imshow(y)
		return y

	def get_bounding_boxes(self, img, orig_img=None):
# 	def get_bounding_boxes(self, img):
		'''
			Find the bounding boxes of the recycling bins
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				boxes - a list of lists of bounding boxes. Each nested list is
                a bounding box in the form of [x1, y1, x2, y2] 
				where (x1, y1) and (x2, y2) are the top left and bottom right 
                coordinate respectively
		'''
		# YOUR CODE HERE
		# smooth out boundaries for
# 		img = self.segment_image(img)
# 		img_test=img.copy()

		erosion_kernel = np.ones((11,11),np.uint8)   # erosion kernel
		img = cv2.erode(img,erosion_kernel,iterations = 1)  # erode image
# 		plt.imshow(img)
# 		plt.show()
# 		dilate_kernel = np.ones((15,15), np.uint8)  # dilate kernel
# 		img = cv2.dilate(img, dilate_kernel, iterations = 1) # dilate image
# 		plt.imshow(img)
#		plt.show()
		
		img = cv2.GaussianBlur(img, (3,3),0)
		# create 15 bounding boxes
		contours,x = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE  )[-2:]
# 		print('here is contour', contours)
# 		print('here is x', x)
		contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)[:5] #largest 3 boxes
		img_to_cont = cv2.drawContours(orig_img, contours_sorted, -1, (0,0,0), 1)
# 		print('contours', len(contours_sorted[0]))
		plt.imshow(img_to_cont)
		plt.show()
		boxes = []
# 		print(contours_sorted)
		for contour in contours_sorted:
# 			print('loop through contours_sorted',contour)
			perimeter = 0.1*cv2.arcLength(contour, True)
			approx_rect = cv2.approxPolyDP(contour, perimeter, True)
			(x, y, w, h) = cv2.boundingRect(approx_rect)
			
			temp_box = [x, y, x+w, y+h]
			
			aspect_ratio = float(h)/float(w)
# 			print('first ratio',aspect_ratio)

# 			cv2.imshow("my box", boxed_img)
# 			cv2.waitKey(0)

			
			
			if aspect_ratio <= 6 and aspect_ratio> 1 and (w*h>=400 and w*h/float(img.shape[0]*img.shape[1]) > 0.02):
# 				print('Here is the ratio',aspect_ratio)
				boxes.append(temp_box)
				boxed_img = cv2.rectangle(orig_img, (x,y),(x+w,y+h),(0,255,0), 2)
				boxed_img=cv2.cvtColor(boxed_img,cv2.COLOR_BGR2RGB)
				plt.imshow(boxed_img)
				plt.show()
				
			
# 		x = np.sort(np.random.randint(img.shape[0],size=2)).tolist()
# 		y = np.sort(np.random.randint(img.shape[1],size=2)).tolist()
# 		boxes = [[x[0],y[0],x[1],y[1]]]
# 		boxes = [[182, 101, 313, 295]]
		return boxes

import os 
if __name__ == '__main__':
	val_image_path = 'C:/Users/s/Google Drive/UCSD GRADUATE CLASSES/2021 WINTER/ECE 276A/pr1/ECE276A_PR1/data/validation/'
# 	filelist = os.path.dirname(os.path.dirname(val_image_path)+'validation')
	filelist = os.listdir(val_image_path)
	for file in filelist:
		print(file)
		if (file.endswith(".jpg") or file.endswith(".png")):
# 			filelist.remove(file)
			print('here is file name', file)
			bindet = BinDetector()
			image = val_image_path+file
			img = cv2.imread(image)
			img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
			orig_img = img.copy()
			segment = bindet.segment_image(img)
			plt.imshow(segment)
			plt.show()
			cont2 = bindet.get_bounding_boxes(segment)
		else:
 			print('here is file name', file)
 			bindet = BinDetector()
 			image = val_image_path+file
 			img = cv2.imread(image)
 			segment = bindet.segment_image(img)
 			plt.imshow(segment)
 			plt.show()
 			cont2 = bindet.get_bounding_boxes(segment)
	contours = bindet.get_bounding_boxes(img)

imgs = segment
erosion_kernel = np.ones((11,11),np.uint8)   # erosion kernel
imgs = cv2.erode(imgs,erosion_kernel,iterations = 1)  # erode image
plt.imshow(imgs)
plt.show()
dilate_kernel = np.ones((15,15), np.uint8)  # dilate kernel
imgs = cv2.dilate(imgs, dilate_kernel, iterations = 1) # dilate image
plt.imshow(imgs)
plt.show()

imgs = cv2.GaussianBlur(imgs, (3,3),0)
# create 15 bounding boxes
contours,x = cv2.findContours(imgs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE  )[-2:]
# 		print('here is contour', contours)
# 		print('here is x', x)
contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)[:5] #largest 3 boxes
img_to_cont = cv2.drawContours(orig_img, contours_sorted, -1, (0,0,0), 1)
# 		print('contours', len(contours_sorted[0]))
plt.imshow(img_to_cont)
plt.show()
boxes = []
	
