import os
import numpy as np
import cv2

import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

# does initial image preprocessing (blurring, remove noise, convert black background white text)
def initial_preprocess(img_path, gaussian_blur = -1):
    # read image in grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)    
    orig = img.copy()
    
    # Do Gaussian Blur in order to remove noise
    if gaussian_blur != -1:
        img = cv2.GaussianBlur(img, (gaussian_blur, gaussian_blur), 0)
    
    # do adaptive thresholding and convert images to binary
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                 cv2.THRESH_BINARY,11,2)

    # convert to black on white 
    img = cv2.bitwise_not(img, img)

    # dilate the gridlines
    #kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],np.uint8)
    #img = cv2.dilate(img, kernel)
    
    # find contours (Sudoku Grid)
    contours,hierarchy = cv2.findContours(img, 1, 2)
    contours = sorted(contours, key=cv2.contourArea, reverse=True) # sort contours by area
    polygon_arr = contours[0:5]  # extract up to 5 sudoku grid at once

    cropped_arr = []
    cropped_arr_orig = []
    bbox_arr = []
    orig_gray = img.copy()
    
    # enumerate to draw rectangle and crop
    for polygon in polygon_arr:
        x,y,w,h = cv2.boundingRect(polygon)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        bbox_arr.append((x,y,w,h))
        cropped_arr.append(orig_gray[y:y+h, x:x+w])
        cropped_arr_orig.append(orig[y:y+h, x:x+w])
    
    return cropped_arr, cropped_arr_orig, bbox_arr
    
# find all the edges that are rectangular and returns them
def find_edges(cropped_arr):
    all_edges = []
    
    for i in range(5):
        # detect edges
        img1 = cropped_arr[i].copy()

        # find contours (Sudoku Grid)
        contours,hierarchy = cv2.findContours(img1, 1, 2)
        contours = sorted(contours, key=cv2.contourArea, reverse=True) # sort contours by area
        polygon = np.array(contours[0]).reshape(-1,2) #take the largest contour region

        sums = [elem[0] + elem[1] for elem in polygon] # X + Y
        diff1 = [elem[0] - elem[1] for elem in polygon] # X - Y

        top_left = np.argmin(sums) # smallest X + Y values
        bottom_right = np.argmax(sums) # largest X + Y values
        bottom_left = np.argmin(diff1) # smallest X - Y values
        top_right = np.argmax(diff1) # largest X - Y values

        # convert to tuple
        edges = [tuple(polygon[top_left]), tuple(polygon[top_right]), 
                 tuple(polygon[bottom_right]), tuple(polygon[bottom_left])]
        all_edges.append(edges)

        # draw edges with gray circle
        #for elem_edge in edges:
        #    cv2.circle(img1, elem_edge, 3, (100,100,255), 2)
    
    return all_edges

# check if there are any valid sudoku
def find_valid_sudoku(all_edges, cropped_arr_orig):
    # filter only valid images
    valid_sudoku_images = list(range(5))
    to_remove = []

    # check if edges are within 50 pixels in the corner.
    for elem in valid_sudoku_images:
        if all_edges[elem][0][0] > 50 or all_edges[elem][0][1] > 50: # top left of image not very close to edge
            to_remove.append(elem)

    valid_sudoku_images = [x for x in valid_sudoku_images if x not in to_remove]  # remove invalid

    img_size_area = [elem.shape[0] * elem.shape[1] for elem in cropped_arr_orig] # find area
    max_area = np.max(img_size_area) # get maximum area
    for elem in valid_sudoku_images:
        if img_size_area[elem] / max_area < 0.5: # less than 50% of max area
            to_remove.append(elem)

    valid_sudoku_images = [x for x in valid_sudoku_images if x not in to_remove] # remove invalid
    
    # if no valid boxes are found just add the first box (most probably correct)
    if len(valid_sudoku_images) == 0:
        valid_sudoku_images.append(0)
        all_edges = []
        h,w = cropped_arr_orig[0].shape
        all_edges = [[(0,0), (h,0), (h,w), (0,w)],]
    
    return valid_sudoku_images

def distance_between(p1, p2):
    """Returns the euclidean distance between two points"""
    a = p2[0] - p1[0]
    b = p2[1] - p1[1]
    return np.sqrt((a ** 2) + (b ** 2))

# warp the image into perspective
def warped_image(all_edges, cropped_arr_orig, valid_sudoku_images):
    all_M = []
    final_proc = []
    
    for idx, elem in enumerate(valid_sudoku_images): 
        # extract edges
        top_left, top_right, bottom_right, bottom_left = all_edges[elem][0], all_edges[elem][1], all_edges[elem][2], all_edges[elem][3]

        # get longest edge of rectangle to warp
        side = max([distance_between(top_left, top_right), 
                     distance_between(top_right, bottom_right),
                     distance_between(bottom_right, bottom_left),
                     distance_between(bottom_left, top_left),
                    ])

        # Describe a square with side of the calculated length, this is the new perspective we want to warp to
        pts2 = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')

        # transforms image from tilted to straight
        pts1 = np.float32(all_edges[elem])

        # M is matrix for transformation
        M = cv2.getPerspectiveTransform(pts1,pts2)
        all_M.append(M)

        # output
        dst = cv2.warpPerspective(cropped_arr_orig[idx],M,(int(side),int(side)))
        final_proc.append(dst)
    
    return final_proc, all_M

# post process the warped image
def post_process_warped(final_proc, gauss_blur = -1):
    final_proc_final = []
    # preproces finalboard
    for elem in final_proc:
        if gauss_blur != -1:
            # gaussian blur
            proc = cv2.GaussianBlur(elem.copy(), (gauss_blur, gauss_blur), 0)
        else:
            proc = elem.copy()
        
        # Adaptive threshold using 11 nearest neighbour pixels
        proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Invert colours, so gridlines have non-zero pixel values.
        # Necessary to dilate the image, otherwise will look like erosion instead.
        proc = cv2.bitwise_not(proc, proc)
        #kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],np.uint8)
        #proc = cv2.dilate(proc, kernel)

        final_proc_final.append(proc)
    return final_proc_final

# find bounding box of digits
def find_bounding_box(inp_img, scan_top_left, scan_bot_right):
    img = inp_img.copy()
    h, w = img.shape

    max_area = 0
    seed_point = (None, None)

    # Loop through the image
    for x in range(scan_top_left[0], scan_bot_right[0]):
        for y in range(scan_top_left[1], scan_bot_right[1]):
            # if current square is white, fill it with grey
            # flood fill converts neighboring pixels with similar color to white
            if img.item(y, x) == 255 and x < w and y < h:  # Note that .item() appears to take input as y, x
                area = cv2.floodFill(img, None, (x, y), 64)
                if area[0] > max_area:  # Gets the maximum bound area which should be the grid, and set start pt
                    max_area = area[0]
                    seed_point = (x, y)

    # Colour everything grey (compensates for features outside of our middle scanning range)
    for x in range(w):
        for y in range(h):
            if img.item(y, x) == 255 and x < w and y < h:
                cv2.floodFill(img, None, (x, y), 64)

    mask = np.zeros((h + 2, w + 2), np.uint8)  # Mask that is 2 pixels bigger than the image (from documentation)

    # Highlight the main feature, fill it with white
    if all([p is not None for p in seed_point]):
        cv2.floodFill(img, mask, seed_point, 255)

    top, bottom, left, right = h, 0, w, 0
    
    # fill those that are grey with black to hide anything that is not the main feature
    for x in range(w):
        for y in range(h):
            if img.item(y, x) == 64:  # Hide anything that isn't the main feature
                cv2.floodFill(img, mask, (x, y), 0)
            
            # Find the bounding parameters (find the last contiguous x and y)
            if img.item(y, x) == 255:
                top = y if y < top else top
                bottom = y if y > bottom else bottom
                left = x if x < left else left
                right = x if x > right else right

    # returns the bounding box of the characters
    return left, top, right, bottom

# scale the digits
def scale_and_centre(img, size, margin=0, background=0):
    """Scales and centres an image onto a new background square."""
    h, w = img.shape

    def centre_pad(length):
        """Handles centering for a given length that may be odd or even."""
        if length % 2 == 0:
            side1 = int((size - length) / 2)
            side2 = side1
        else:
            side1 = int((size - length) / 2)
            side2 = side1 + 1
        return side1, side2

    def scale(r, x):
        return int(r * x)
    
    # if height greater than width, need to scale appropriately
    if h > w:
        t_pad = int(margin / 2)
        b_pad = t_pad
        ratio = (size - margin) / h
        w, h = scale(ratio, w), scale(ratio, h)
        l_pad, r_pad = centre_pad(w)
    else:
        l_pad = int(margin / 2)
        r_pad = l_pad
        ratio = (size - margin) / w
        w, h = scale(ratio, w), scale(ratio, h)
        t_pad, b_pad = centre_pad(h)

    img = cv2.resize(img, (w, h))
    img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
    return cv2.resize(img, (size, size))

# split the images into cells
def process_to_cell(final_proc):
    # split into 9x9 grids
    board_arr = []
    for board in final_proc:
        one_side_length = board.shape[0] / 9
        temp = []
        for i in range(9):
            lc = int(i * one_side_length)
            for j in range(9):
                rc = int(j * one_side_length)
                # currbox raw just take the raw pixels
                currbox = board[lc:int(lc+one_side_length), rc:int(rc+one_side_length)]
                

                margin = int(currbox.shape[0] / 2.5)    
                # get bounding box, input the scanning region
                left, top, right, bottom = find_bounding_box(currbox, [margin, margin], 
                                                             [currbox.shape[0] - margin, currbox.shape[1] - margin]) 
                # there is no object in the current box (set everything to black)
                if left == top and right == bottom:
                    currbox = np.zeros((28,28))
                # crop the digits in the image
                else:
                    currbox = currbox[top:bottom + 2, left:right + 2]
                    # center and pad the images
                    currbox = scale_and_centre(currbox, 28,4)
                    # binary tresholding
                    currbox = cv2.threshold(currbox,80, 255, cv2.THRESH_BINARY)[1]

                # add them to the array row wise
                temp.append(currbox)            
        board_arr.append(temp)

    board_arr = np.array(board_arr)
    return board_arr

def load_model(model_path = 'digit_recognizer_best.h5'):    
    # load from saved pre-trained model (LeNet-5 Architecture modified)
    model = tf.keras.models.load_model(model_path)
    return model

# convert into numpy array
def convert_board_to_numpy(board_arr, model):
    sudoku_board = []

    for i in range(board_arr.shape[0]):
        temp = []
        for j in range(board_arr.shape[1]):
            if len(np.where(board_arr[i][j][7:17, 7:17].ravel() == 0)[0])/100 > 0.8:  # percentage of 0 which is black
                temp.append(0)
            else:
                curr_box = board_arr[i][j].reshape(1,28,28,1) / 255
                y_pred = np.argmax(model.predict(curr_box), axis=-1)[0]
                if y_pred == 0:
                    curr_pred = model.predict(curr_box).ravel()
                    maxs_idx = curr_pred.argsort()[-2:]
                    y_pred = maxs_idx[0]
                temp.append(y_pred)

        sudoku_board.append(temp)
    
    return sudoku_board
