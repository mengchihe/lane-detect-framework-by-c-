#include "common.h"
#include "cnn_hmc/include/cnn_hmc.h"
#include "semantic_seg.h"
#define video_input

int g_cur_frame_num;

void main(int argc, char *argv[])
{
	tensor_c<int> input_tensor(1,2,3,4);
	int width = ORI_IMG_WIDTH;
	int height = ORI_IMG_HEIGHT;
	int i, j, k, temp;
	clock_t start, finish;

#ifdef video_input
	semantic_seg_c<float> semantic_seg("mat");
	semantic_seg.create_net();
	vector<string> file_name = get_files("E:\\video\\video_data\\19_01_31\\ours");
	string out_folder = "E:\\video\\video_data\\19_01_31\\temp\\";
	int start_file_num = 0;
	VideoCapture video_in;
	Mat cur_frame;
	Mat cur_yuv(height, width, CV_8UC3, Scalar(0, 0, 0));
	namedWindow("out");
	char show_str[1024];
	CvFont font_show;
	cvInitFont(&font_show, CV_FONT_HERSHEY_COMPLEX, 1.0, 1.0, 0, 1, 8);
	int file_num = file_name.size();
	if (file_num == 0)
	{
		printf("no input file!!!!!!!\n");
		system("pause");
	}
	for (int file_cnt = start_file_num; file_cnt < file_num; file_cnt++)
	{
		printf("read_video: %s\n", file_name[file_cnt].c_str());
		video_in.open(file_name[file_cnt].c_str());
		string out_file = out_folder;
		size_t filename_pos = file_name[file_cnt].find_last_of("\\");
		size_t dot_pos = file_name[file_cnt].find_last_of(".");
		out_file += file_name[file_cnt].substr(filename_pos + 1, dot_pos - filename_pos - 1);
		out_file += "_out.avi";
		VideoWriter video_out(out_file.c_str(), CV_FOURCC('D', 'I', 'V', 'X'), 25.0, cvSize(1280, 720));
		g_cur_frame_num = 0;
		while (video_in.read(cur_frame))
		{
			Mat *frame_pt = &cur_frame;
			Mat output_frame(CNN_INPUT_HEIGHT, CNN_INPUT_WIDTH, CV_8UC3, Scalar(0, 0, 0));

			start = clock();
			semantic_seg.semantic_seg_frame(frame_pt, &output_frame);
			finish = clock();
			cout << finish - start << ": frame time" << endl;

			imshow("out", output_frame);
			cvWaitKey(1);
			video_out << output_frame;
			g_cur_frame_num++;
			printf("%d\n", g_cur_frame_num);
		}
	}
#else
#endif

}