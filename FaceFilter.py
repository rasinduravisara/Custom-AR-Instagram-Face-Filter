import cv2
import numpy as np
import dlib
from math import hypot

cap = cv2.VideoCapture(0) #Camera Input
glassImg = cv2.imread("Glass.png") #Glass Image
glsMask = cv2.imread("GlassMaskFill.png") #Glass Image Mask

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Import Landmark Detector

while True:
	_,frame = cap.read() # Read Camera Input
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = detector(gray)

	# Select the Part that needs to be replaced using landmarks

	for face in faces:
		landmarks = predictor(gray, face)

		eyeMid = (landmarks.part(28).x, landmarks.part(28).y)
		eyeLeft = (landmarks.part(36).x, landmarks.part(36).y)
		eyeRight = (landmarks.part(45).x, landmarks.part(45).y)
		eyeDown = (landmarks.part(28).x, landmarks.part(28).y)

		# cv2.circle(frame, eyeMid, 3 ,(255,0,0), -1)
		# cv2.circle(frame, eyeLeft, 3 ,(255,0,0), -1)
		# cv2.circle(frame, eyeRight, 3 ,(255,0,0), -1)

		eyeWidth = int(hypot(eyeLeft[0] - eyeRight [0], eyeLeft[1] - eyeRight [1])) + 15
		eyeHeight = int(eyeWidth * 0.22)

		topLeft = (int(eyeMid[0] - eyeWidth/2), int(eyeMid [1] - eyeHeight/2 ))
		bottomRight = (int(eyeMid[0] + eyeWidth/2) , int(eyeMid [1] + eyeHeight/2))

		# cv2.rectangle(frame,(int(eyeMid[0] - eyeWidth/2), 
		# 					int(eyeMid [1] - eyeHeight/2 )),
		# 					(int(eyeMid[0] + (eyeWidth/2)) , 
		# 					int(eyeMid [1] + (eyeHeight/2))),
		# 					(0,255,0), 2)


		eyeGlass = cv2.resize(glassImg,(eyeWidth, eyeHeight)) #Resize Glass image
		mask = cv2.resize(glsMask,(eyeWidth, eyeHeight)) #Resize Mask
		eyeGlassGray = cv2.cvtColor(eyeGlass,cv2.COLOR_BGR2GRAY)
		MaskGray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
		_,filterMask = cv2.threshold(MaskGray,50,255, cv2.THRESH_BINARY) 
		filterMaskInvert = cv2.bitwise_not(filterMask)
		eyeGlass = cv2.bitwise_and(eyeGlass,eyeGlass, mask = filterMaskInvert)

		filterArea = frame[topLeft[1] : topLeft[1] + eyeHeight, topLeft[0] : topLeft[0] + eyeWidth]
		filterArea_withoutFilter = cv2.bitwise_and(filterArea,filterArea, mask = filterMask)
		
		filterFinal = cv2.add(filterArea_withoutFilter, eyeGlass)

		frame[topLeft[1] : topLeft[1] + eyeHeight, topLeft[0] : topLeft[0] + eyeWidth] = filterFinal

		# cv2.imshow("filterArea", filterArea)
		# cv2.imshow("finalfilter", filterFinal)

	cv2.imshow("Frame",frame)
	# cv2.imshow("EyeGlass",eyeGlass)
	# cv2.imshow("mask",filterMask)
	# cv2.imshow("AreaWithotMask", filterArea_withoutFilter)
	cv2.waitKey(1)
