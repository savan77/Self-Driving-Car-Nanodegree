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

def complete_lines_new(lines, imshape):
    
    slope_l,slope_r,l_x,l_y,r_x,r_y,count_l,count_r=0,0,0,0,0,0,0,0
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y2-y1)/(x2-x1)
            if slope >0.1:
                slope_l += slope 
                l_x += (x1+x2)/2
                l_y += (y1+y2)/2
                count_l+= 1
            elif slope < -0.2:
                slope_r += slope
                r_x += (x1+x2)/2
                r_y += (y1+y2)/2
                count_r+= 1
  
    if count_l>0:
        avg_slope_left = slope_l/count_l
        avg_leftx = l_x/count_l
        avg_lefty = l_y/count_l
        xb_l = int(((int(0.97*imshape[0])-avg_lefty)/avg_slope_left) + avg_leftx)
        xt_l = int(((int(0.61*imshape[0])-avg_lefty)/avg_slope_left)+ avg_leftx)

    
    if count_r>0:
        avg_slope_right = slope_r/count_r
        avg_rightx = r_x/count_r
        avg_righty = r_y/count_r
        xb_r = int(((int(0.97*imshape[0])-avg_righty)/avg_slope_right) + avg_rightx)
        xt_r = int(((int(0.61*imshape[0])-avg_righty)/avg_slope_right)+ avg_rightx)
 
    return [[xt_l, int(0.61*imshape[0]), int(1.04*xb_l), int(1*imshape[0])],[xt_r, int(0.61*imshape[0]), int(0.85*xb_r), int(1*imshape[0])]]

def complete_polyfit(lines):
	def split_lines(lines):
		one = []
		two = []
		for line in lines:
			for x1,y1,x2,y2 in line:
				slope = (y2-y1)/(x2-x1)
				if slope > 0.1:
					one.append([x1,y1,x2,y2])
				elif slope < -0.2:
					two.append([x1,y1,x2,y2])
		return [one, two]

	def create_lists(x,y):
		l = list(zip(x,y))
		prev = l[0]
		out = []
		for i in l[1:]:
			# print(prev)
			# print(i)
			prev = i
			out.append([int(prev[0]),int(prev[1]),int(i[0]),int(i[1])])
		return out


	lines_new = split_lines(lines)
	out_list =[]
	for splitted_lines in lines_new:
		x = []
		y = []
		
		for x1,y1,x2,y2 in splitted_lines:
			x += [x1,x2]
			y += [y1,y2]
		f = np.poly1d(np.polyfit(x,y,1))
		x_out = list(range(min(x), max(y)))
		y_out = f(list(range(min(x), max(y))))
		out_list.extend(create_lists(x_out, y_out))

	return out_list



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
	vertices = np.array([[(120,imshape[0]),(int(imshape[1]/2)-50,int(imshape[0]/2)+60), (int(imshape[1]/2)+50, int(imshape[0]/2+60)), (imshape[1]-75,imshape[0])]], dtype=np.int32)
	cv2.fillPoly(mask, vertices, ignore_mask_color)
	masked_edges = cv2.bitwise_and(edges, mask)

	rho = 1 # distance resolution in pixels of the Hough grid
	theta = np.pi/180 # angular resolution in radians of the Hough grid
	threshold = 15    # minimum number of votes (intersections in Hough grid cell)7
	min_line_length = 10 #minimum number of pixels making up a line15
	max_line_gap = 3    # maximum gap in pixels between connectable line segments2
	line_image = np.copy(frame)*0 # creating a blank to draw lines on

	lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
	                            min_line_length, max_line_gap)
	# lines = [complete_lines_new(lines, imshape)]
	# for line in lines:
	for line in lines:
		for x1,y1,x2,y2 in line:
			cv2.line(line_image,(x1,y1),(x2,y2),(0,0,255),7)

	frame_copy = frame.copy()
	frame_c = frame.copy()
	cv2.polylines(frame_copy, [vertices.reshape((-1,1,2))], True, (0,255,0), 5)

	# Create a "color" binary image to combine with line image
	color_edges = np.dstack((edges, edges, edges)) 

	# Draw the lines on the edge image
	lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0) 
	lines_org = cv2.addWeighted(frame_c, 0.8, line_image, 1,0)
	return lines_edges, frame_copy, lines_org


cap = cv2.VideoCapture(sys.argv[1])

FRAME_COUNT = 0
while True:
	FRAME_COUNT += 1
	print("Frame Number: ", FRAME_COUNT)
	ret, frame = cap.read()
	lines_edges, frame_copy, lines_org = simple_line_detect(frame)
	

	

	draw_window("Original", frame)
	draw_window("Edges", lines_edges)
	draw_window("RoI", frame_copy)
	draw_window("LineOverlay", lines_org)
	# roi = draw_window("RoI")
	# cv2.imshow("output"/
