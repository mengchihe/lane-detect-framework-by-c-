#include "common.h"


vector<string> get_files(string path)
{
	vector<string> file_name;
	intptr_t lf;
	_finddata_t file;
	string p;
	if ((lf = _findfirst(p.assign(path).append("\\*").c_str(), &file)) == -1)
	{
		cout << path << "not found!!!!!!!!!!!!!" << endl;
	}
	else
	{
		while (_findnext(lf, &file) == 0)
		{
			if (strcmp(file.name, ".") == 0 || strcmp(file.name, "..") == 0)
				continue;
			file_name.push_back(p.assign(path).append("\\").append(file.name));
		}
	}
	_findclose(lf);
	return file_name;
}

void mat_to_bgr(byte *yuv_img, Mat *mat_img, int width, int height)
{
	int i, j;
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			if (i == 100 && j == 100)
			{
				int qqq = 3;
			}
			yuv_img[i * width + j] = (*mat_img).data[3 * (i*width + j) + 0];
			yuv_img[width * height + i * width + j] = (*mat_img).data[3 * (i*width + j) + 1];
			yuv_img[2 * width * height + i * width + j] = (*mat_img).data[3 * (i*width + j) + 2];
		}
	}
}

void bgr_to_mat(Mat *mat_img, byte *yuv_img, int width, int height)
{
	int i, j;
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			(*mat_img).data[3 * (i * width + j) + 0] = yuv_img[i * width + j];
			(*mat_img).data[3 * (i * width + j) + 1] = yuv_img[width * height + i * width + j];
			(*mat_img).data[3 * (i * width + j) + 2] = yuv_img[2 * width * height + i * width + j];
		}
	}
}

void img_scaling(byte *img_in, byte *img_out, string method)
{
	int input_width = ORI_IMG_WIDTH;
	int input_height = ORI_IMG_HEIGHT;
	int output_width = CNN_INPUT_WIDTH;
	int output_height = CNN_INPUT_HEIGHT;

	if (method == "copy")
	{
		memcpy(img_out, img_in, 3 * input_width * input_height * sizeof(byte));
		return;
	}
	int i, j, k;

	int v_step, h_step;
	v_step = input_height * 4096 / output_height;
	h_step = input_width * 4096 / output_width;
	int phase_v, phase_h, inter_phase_v, inter_phase_h;
	for (k = 0; k < 3; k++)
	{
		byte *cur_in_channel = img_in + input_width * input_height * k;
		byte *cur_out_channel = img_out + output_width * output_height * k;
		phase_v = -(v_step - 4096) / 2;
		for (i = 0; i < output_height; i++)
		{
			inter_phase_v = ((phase_v - (phase_v / 4096) * 4096) * 32) / 4096;
			phase_h = -(h_step - 4096) / 2;
			for (j = 0; j < output_width; j++)
			{
				if (k == 1 && i == 9 && j == 9)
				{
					int qqq = 3;
				}
				inter_phase_h = ((phase_h - (phase_h / 4096) * 4096) * 32) / 4096;
				int cur_v_pos = max(0, min(input_height - 2, phase_v / 4096));
				int cur_h_pos = max(0, min(input_width - 2, phase_h / 4096));
				if (method == "bilinear")
				{
					byte top_left = cur_in_channel[cur_v_pos * input_width + cur_h_pos];
					byte top_right = cur_in_channel[cur_v_pos * input_width + cur_h_pos + 1];
					byte bot_left = cur_in_channel[(cur_v_pos + 1) * input_width + cur_h_pos];
					byte bot_right = cur_in_channel[(cur_v_pos + 1) * input_width + cur_h_pos + 1];
					cur_out_channel[i*output_width + j] = ((32 - inter_phase_v) * (32 - inter_phase_h) * top_left
						+ (32 - inter_phase_v) * inter_phase_h * top_right
						+ inter_phase_v * (32 - inter_phase_h) * bot_left
						+ inter_phase_v * inter_phase_h * bot_right
						) / 32 / 32;
				}
				else if (method == "nearest")
				{
					cur_out_channel[i*output_width + j] =
						cur_in_channel[cur_v_pos * input_width + cur_h_pos];
				}
				phase_h += h_step;
			}
			phase_v += v_step;
		}
	}



}