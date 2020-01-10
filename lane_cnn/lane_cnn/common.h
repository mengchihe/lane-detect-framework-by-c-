/*include all basic math and read/write function*/
#ifndef COMMON_H
#define COMMON_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <io.h>
#include <stdio.h>
#include <fstream>
#include <boost/smart_ptr.hpp>
#include <boost/make_shared.hpp>


#define _CRT_SECURE_NO_WARNINGS
typedef unsigned char byte;
using namespace std;
using namespace cv;

#define max(a, b)  (((a) > (b)) ? (a) : (b))
#define min(a, b)  (((a) < (b)) ? (a) : (b))
#define ORI_IMG_WIDTH 1280
#define ORI_IMG_HEIGHT 720
#define CNN_INPUT_WIDTH 512
#define CNN_INPUT_HEIGHT 256


vector<string> get_files(string path);
void mat_to_bgr(byte *yuv_img, Mat *mat_img, int width, int height);
void bgr_to_mat(Mat *mat_img, byte *yuv_img, int width, int height);
void img_scaling(byte *img_in, byte *img_out, string method);

#endif