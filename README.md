# AGCWD
This is a C++ implementation code using OpenCV libraryfor the paper "Efficient Contrast Enhancement Using Adaptive Gamma Correction With Weighting Distribution" at: https://ieeexplore.ieee.org/abstract/document/6336819/.


 Inputs:
          
          -input_image: can be either gray image or colorful image
          
          -parameter  : (optional) weighting parameter for the histogram
           can be [0,1]. Default is 0.5
   
   Output:
   
          -enhanced_image: the result image after applying AGCWD contrast enhancement
