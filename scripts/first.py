import cv2
import numpy as np
from math import sqrt
import sys

def draw_window(window_name, frame):
	cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
	cv2.imshow(window_name, frame)
	cv2.waitKey(1)

def euclidian(a,b,c,d):
	return sqrt((c - a)**2 + (d-b)**2)
	

def join_lines(lines):
	prev = 0,0
	f = 0,0
	first = True
	new_lines = []
	for line in lines:
		
		if first:
			first = False
		else:
			new_lines.append([prev[0]+1, prev[1]+1,line[0]+1,line[1]+1])
		prev = line[2], line[3]	
	return new_lines
		
def draw_parallel(lines):
	pass

def complete_lines(lines,imshape):
	one_side = []
	other_side = []
	for line in lines:
		for x1,y1,x2,y2 in line:
			# print(x1,y1,x2,y2)
			if y2-y1 < 0:
				one_side.append([x1,y1,x2,y2])
			else:
				other_side.append([x1,y1,x2,y2])
	n1 = join_lines(one_side)
	n2 = join_lines(other_side)
	return one_side + other_side + n1 + n2

def simple_line_detect(frame):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray,(5, 5), 0)

	low_threshold = 50
	high_threshold = 150
	edges = cv2.Canny(blur, low_threshold, high_threshold)

	mask = np.zeros_like(edges)   
	ignore_mask_color = 255   

	# This time we are defining a four sided polygon to mask
	imshape = frame.shape
	vertices = np.array([[(110,imshape[0]),(int(imshape[1]/2)-35,int(imshape[0]/2)+45), (int(imshape[1]/2)+40, int(imshape[0]/2+45)), (imshape[1]-75,imshape[0])]], dtype=np.int32)
	cv2.fillPoly(mask, vertices, ignore_mask_color)
	masked_edges = cv2.bitwise_and(edges, mask)

	rho = 2 # distance resolution in pixels of the Hough grid
	theta = np.pi/180 # angular resolution in radians of the Hough grid
	threshold = 7     # minimum number of votes (intersections in Hough grid cell)
	min_line_length = 15 #minimum number of pixels making up a line
	max_line_gap = 2    # maximum gap in pixels between connectable line segments
	line_image = np.copy(frame)*0 # creating a blank to draw lines on

	lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
	                            min_line_length, max_line_gap)
	lines = complete_lines(lines,imshape)

	for x1,y1,x2,y2 in lines:
		if y2-y1 < 0:
			cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),11)

	frame_copy = frame.copy()
	cv2.polylines(frame_copy, [vertices.reshape((-1,1,2))], True, (0,255,0), 5)

	# Create a "color" binary image to combine with line image
	color_edges = np.dstack((edges, edges, edges)) 

	# Draw the lines on the edge image
	lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0) 
	return lines_edges, frame_copy


cap = cv2.VideoCapture(sys.argv[1])

FRAME_COUNT = 0
while True:
	FRAME_COUNT += 1
	print("Frame Number: ", FRAME_COUNT)
	ret, frame = cap.read()
	lines_edges, frame_copy = simple_line_detect(frame)
	

	

	draw_window("Original", frame)
	draw_window("Edges", lines_edges)
	draw_window("RoI", frame_copy)
	# roi = draw_window("RoI")
	# cv2.imshow("output"/
