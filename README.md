# lane-detect-framework-by-c-
a semantic segment network for lane detect, realized by c++, and use opencv to read input video
the project is based on visual studio 2015, only realize forward prop with float data type

all source codes are in lane_cnn, and components of neural network are in cnn_hmc
the model file is model.txt, weights for all layers are stored one by one in weight.txt
if you want to run the video, you should put all videos in one folder and put its path in line 19 in main.cpp
