# Sudoku-Solver-Images
#### Detects and Solve Sudoku Board from Input Image
#### Link to Application (Deployed to Heroku): https://sudoku-solver-images.herokuapp.com/

### Examples:
<p align="center"> <img src=https://github.com/jsantoso2/Sudoku-Solver-Images/blob/master/Screenshots/Capture2.JPG width="700"></p>
<p align="center">Demo for Images<p align="center">

### Dataset
- MNIST Dataset
- Chars74k Dataset (http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/)
  - Only took 0-9 digits images

### Tools/Framework Used
- Image Preprocessing: OpenCV2
- ML models: CNN in Keras
- Web Application: Streamlit (https://www.streamlit.io/)
- Deployment: Heroku
- Data Cleaning/Manipulation: Python 

### Procedure
- Image Preprocessing
  - Gaussian Blurring (if used)
  - Adaptive Thresholding
  - Find Contours of rectangle for sudoku board
  - Crop and Warp perspective
  - Split the warped image into 9x9 grids
- Prediction
  - From the 9x9 grids, use CNN to predict digits
  - CNN pretrained from MNIST and Chars74k dataset with 98.9% accuracy on validation set.
- Sudoku Solver: Utilizes Norvig’s Algorithm (https://norvig.com/sudoku.html)
- Create Streamlit Web App

### Streamlit Web Application Screenshots
<p align="center"> <img src=https://github.com/jsantoso2/Sudoku-Solver-Images/blob/master/Screenshots/Capture.JPG height="600"></p>
<p align="center">App Interface 1<p align="center">

### References/Inspirations:
- https://becominghuman.ai/image-processing-sudokuai-opencv-45380715a629
- https://github.com/aakashjhawar/SolveSudoku
- https://github.com/ibhanu/sudoku-solver

### Final Notes:
- To see more details, please see notes.docx for all my detailed notes
