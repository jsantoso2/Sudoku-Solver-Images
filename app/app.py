import streamlit as st
import os
import numpy as np
import cv2
from PIL import Image

from image_process import *
from sudoku_solver import *


def main():
    st.set_option('deprecation.showfileUploaderEncoding', False)
    
    # Sidebar Components
    sidebar_html_style = """
        <style> 
        .sidebar-content, .block-container {
            margin: 0;
            padding: 0.5rem 0.5rem 0.5rem 0.5rem !important;
            font-family: Arial, Helvetica, sans-serif;
        }
         
        textarea{
            min-height:250px !important;
        }
        </style>
    """
    st.sidebar.markdown(sidebar_html_style, unsafe_allow_html=True)    
    st.sidebar.subheader("Choose Image to Get Started!")
    radio_option = st.sidebar.radio('Please choose any of the following options',('Choose image from examples','Upload your own image'))
    
    # Add a slider for gaussian blur moved to bottom:
    # sidebar text area moved to bottom after reading images
    
    
    # Main Components
    st.title("Welcome to Image Sudoku Solver!")
    st.write("This Project is inspired by [Aditi Jain's Sudoku Solver](https://becominghuman.ai/image-processing-sudokuai-opencv-45380715a629).")
    st.write("And also by [Aakash Jhawar's Sudoku Solver](https://medium.com/@aakashjhawar/sudoku-solver-using-opencv-and-dl-part-1-490f08701179).")
    
    st.subheader("How does it work?")
    st.markdown("<p>Uses image preprocessing techniques with OpenCV2 and CNN trained on Chars74k dataset and MNIST to recognize digits. \
                         Then utilizes Norvig's Algorithm to solve the sudoku. <p>", unsafe_allow_html=True)
     
    if radio_option == 'Choose image from examples':
        st.subheader("Choose Example Image")
        example_image_option = st.selectbox('Please which example image to display', ('Image1', 'Image2', 'Image3', 'Image4'))
        
        if example_image_option == 'Image1':
            img_path = 'examples/image1.jpg'
        elif example_image_option == 'Image2':
            img_path = 'examples/image2.jpg'
        elif example_image_option == 'Image3':
            img_path = 'examples/image3.jpg'
        elif example_image_option == 'Image4':
            img_path = 'examples/image4.jpg'
        
        st.subheader("Initial Image")
        image = Image.open(img_path)
        st.image(image, caption='Original Image', width = 400)
        
            
    else:
        st.subheader("Input Image File")
        uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg", "JPG", "PNG", "JPEG"])
        img_path = 'examples/image1.jpg'
        st.subheader("Initial Image")
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', width = 400)
            image.save('temp.jpg')
            img_path = 'temp.jpg'
        else:
            image = Image.open(img_path)
            st.image(image, caption='Original Image', width = 400)            
    
    # gaussian_blur_treshold
    gaussian_blur_tresh = -1
    st.sidebar.subheader("Adjust Gaussian Blur for Image Processing")
    st.sidebar.warning('For schreenshots keep at -1  \n (No blurring)!')
    st.sidebar.markdown("<p>If sudoku board is not detected OR preprocessed image is too noisy, adjust this parameter!<p>", unsafe_allow_html=True)
    blur_slider = st.sidebar.slider('Select a values for Gaussian Blur Kernel Size: ', -1, 13, gaussian_blur_tresh, step = 2)
    gaussian_blur_tresh = int(blur_slider)
    
    # does initial preprocessing of images and obtain up to 5 boxes for sudoku
    # orig is for the non-preprocessed image
    cropped_arr, cropped_arr_orig, bbox_arr = initial_preprocess(img_path, gaussian_blur_tresh)

    # find edges of each of the boxes
    all_edges = find_edges(cropped_arr)

    # eliminate invalid sudoku boxes
    valid_sudoku_images = find_valid_sudoku(all_edges, cropped_arr_orig)

    # take out unneded boxes
    cropped_arr = [x for idx, x in enumerate(cropped_arr) if idx in valid_sudoku_images]
    cropped_arr_orig = [x for idx, x in enumerate(cropped_arr_orig) if idx in valid_sudoku_images]

    # warp and transform on original image (not preprocessed)
    final_proc, all_M = warped_image(all_edges, cropped_arr_orig, valid_sudoku_images)

    # now preprocess the cropped and warped image
    final_proc = post_process_warped(final_proc, gaussian_blur_tresh)

    # split the board into cells and clean the images
    board_arr = process_to_cell(final_proc)

    # load model
    model = load_model('digit_recognizer_best.h5')

    # convert board into numpy array
    sudoku_board = convert_board_to_numpy(board_arr, model)
    
    # display processed image
    init_proc = []
    init_capt = []
    for len_board in range(len(sudoku_board)):
        init_proc.append(final_proc[len_board])
        init_capt.append('Board' + str(len_board + 1))
    
    st.subheader('Preprocessed Image (After Gaussian Filtering)')
    if len(sudoku_board) == 1:
        st.image(init_proc, init_capt, width = 400)
    else:
        st.image(init_proc, init_capt, width = 300) 
    
    
    # Manual Fixing for 3rd example
    if img_path == 'examples/image3.jpg' and len(sudoku_board) > 1:
        sudoku_board[0][4] = 8
        sudoku_board[0][27] = 1
        sudoku_board[0][37] = 8
        sudoku_board[0][43] = 1
        sudoku_board[0][49] = 1
        sudoku_board[0][66] = 8
        sudoku_board[0][77] = 1
        
        sudoku_board[1][12] = 3        
        sudoku_board[1][16] = 8
        sudoku_board[1][23] = 8
        sudoku_board[1][25] = 2
        sudoku_board[1][33] = 3
        sudoku_board[1][34] = 0
        sudoku_board[1][39] = 0
        sudoku_board[1][37] = 0
        sudoku_board[1][43] = 0
        sudoku_board[1][45] = 8
        sudoku_board[1][50] = 0
        sudoku_board[1][60] = 2
        sudoku_board[1][63] = 0
        sudoku_board[1][67] = 0
        sudoku_board[1][75] = 0
        sudoku_board[1][77] = 0
        sudoku_board[1][80] = 1
       


    
    # sidebar for text area to modify array
    st.sidebar.subheader("Please Modify Grid if Incorrect!")
    grid_text = []
    for idx in range(len(sudoku_board)):
        grid_text_individual = st.sidebar.text_area('Board' + str(idx + 1), value=str(np.array(sudoku_board[idx]).reshape(9,9)), max_chars=200)
        grid_text.append(grid_text_individual)
    
    # preprocess again from gridtext and store in new board
    board_proc_all = []
    for grid in grid_text:
        board_proc = []
        for elem in grid:
            if elem == ' ':
                pass
            elif elem in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                board_proc.append(elem)
        board_proc_all.append(np.array(board_proc).astype(np.uint8))
        
    
    for idx in range(len(sudoku_board)):
        sudoku_board[idx] = board_proc_all[idx]
        
    
    # display recognized images
    recognized_img = []
    recognized_capt = []
    for len_board in range(len(sudoku_board)):
        # read blank sudoku image and prepare to write output
        blank = cv2.imread('blank_grid.jpg')
        blank = cv2.resize(blank, (270,270))
        counter = 0
        for i in range(9):
            for j in range(9):
                if str(sudoku_board[len_board][counter]) != '0':
                    cv2.putText(blank, str(sudoku_board[len_board][counter]), (6 + (j*30), 25 + (i*30)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
                counter += 1
        recognized_img.append(blank)
        recognized_capt.append('Board' + str(len_board + 1) + ' Recognized')
    
    st.subheader('Recognized Board Images (Displayed on sidebar to modify!)')
    if len(sudoku_board) == 1:
        st.image(recognized_img, recognized_capt, width = 400)
    else:
        st.image(recognized_img, recognized_capt, width = 300) 
        

    st.subheader('Solved Sudoku Board Images')
    final_img = []
    final_capt = []
    # iterate through all sudoku board
    for idx,elem in enumerate(sudoku_board):
        a_str = ''.join(str(x) for x in elem)
        solution = main_sudoku(a_str)
        # No solution Available
        if solution == False:
            st.error('No Solution Available, Please adjust gaussian blur OR Modify board in Sidebar!')
        else:
            solution = list(solution.values())
            blank = cv2.imread('blank_grid.jpg')
            blank = cv2.resize(blank, (270,270))
            counter = 0
            for i in range(9):
                for j in range(9):
                    # if not already there previously
                    if sudoku_board[idx][counter] != 0:
                        cv2.putText(blank, str(solution[counter]), (6 + (j*30), 25 + (i*30)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    else:
                        cv2.putText(blank, str(solution[counter]), (6 + (j*30), 25 + (i*30)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
                    counter += 1
            final_img.append(blank)
            final_capt.append('Board' + str(idx+1) + ' Solution')
            
    if len(sudoku_board) == 1:
        st.image(final_img, final_capt, width = 400)
    else:
        st.image(final_img, final_capt, width = 300)         
     
if __name__ == '__main__':
    main()