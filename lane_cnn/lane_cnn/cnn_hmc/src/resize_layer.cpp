#include "../include/resize_layer.h"

template <typename data_type>
resize_layer_c<data_type>::resize_layer_c(ifstream &model_file, vector<tensor_c<data_type>> net_tensors)
{
	type_ = "Resize";
	scaling_method_ = "double";
	string cur_input;
	model_file >> cur_input;
	name_ = cur_input;
	model_file >> cur_input;
	bot_tensor_names_.push_back(cur_input);
	model_file >> cur_input;
	top_tensor_name_ = cur_input;

	int param_read_success[2] = { 0 };
	for (int k = 0; k < 2; k++)
	{
		model_file >> cur_input;
		if (strncmp(cur_input.c_str(), "h_ratio=", 8) == 0)
		{
			h_ratio_ = atof(cur_input.substr(8, cur_input.length() - 1).c_str());
			param_read_success[0] = 1;
		}
		else if (strncmp(cur_input.c_str(), "w_ratio=", 8) == 0)
		{
			w_ratio_ = atof(cur_input.substr(8, cur_input.length() - 1).c_str());
			param_read_success[1] = 1;
		}
		else
		{
			printf("resize_layer.cpp: layer = %s, param = %s, unknown parameter!!!!!!!!!!!!!!!!!\n", name_.c_str(), cur_input.c_str());
			system("pause");
		}
	}
#ifdef DEBUG
	if (param_read_success[0] == 0)
	{
		printf("resize_layer.cpp: layer = %s, param = k, undefined parameter!!!!!!!!!!!!!!!!!\n", name_.c_str());
		system("pause");
	}
	if (param_read_success[1] == 0)
	{
		printf("resize_layer.cpp: layer = %s, param = c, undefined parameter!!!!!!!!!!!!!!!!!\n", name_.c_str());
		system("pause");
	}
#endif

	/*find bot tensor*/
	tensor_c<data_type> *bot_tensor = 0;
	for (int k = 0; k < net_tensors.size(); k++)
	{
		if (net_tensors[k].get_name() == bot_tensor_names_[0])
		{
			bot_tensor = &(net_tensors[k]);
			break;
		}
	}
#ifdef DEBUG
	if (bot_tensor == 0)
	{
		cout << "resize_layer.cpp: can't find layer's bot tensor!!!!!!!!!!\n" << endl;
		system("pause");
	}
#endif
	vector<int> bot_shape = bot_tensor->get_shape();
	top_shape_.push_back(bot_shape[0]);
	top_shape_.push_back(bot_shape[1]);
	top_shape_.push_back(bot_shape[2] / h_ratio_);
	top_shape_.push_back(bot_shape[3] / w_ratio_);
	output_height_ = bot_shape[2] / h_ratio_;
	output_width_ = bot_shape[3] / w_ratio_;
}

template <typename data_type>
resize_layer_c<data_type>::resize_layer_c(tensor_c<data_type> *bot_tensor, int output_height, int output_width, string scaling_method, string name)
{
	output_height_ = output_height;
	output_width_ = output_width;
	scaling_method_ = scaling_method;
	name_ = name;
	bot_tensor_names_.push_back(bot_tensor->get_name());
	top_tensor_name_ = name;

	vector<int> bot_shape = bot_tensor->get_shape();
	top_shape_.push_back(bot_shape[0]);
	top_shape_.push_back(bot_shape[1]);
	top_shape_.push_back(output_height);
	top_shape_.push_back(output_width);
}

template <typename data_type>
resize_layer_c<data_type>::~resize_layer_c()
{

}

template <typename data_type>
void resize_layer_c<data_type>::forward(vector<tensor_c<data_type>> *net_tensors)
{
	/*find bot tensor*/
	tensor_c<data_type> *bot_tensor = 0;
	for (int k = 0; k < net_tensors->size(); k++)
	{
		if ((*net_tensors)[k].get_name() == bot_tensor_names_[0])
		{
			bot_tensor = &((*net_tensors)[k]);
			break;
		}
	}
#ifdef DEBUG
	if (bot_tensor == 0)
	{
		cout << "resize_layer.cpp: can't find layer's bot tensor!!!!!!!!!!\n" << endl;
		system("pause");
	}
#endif
	/*find top tensor*/
	tensor_c<data_type> *top_tensor = 0;
	for (int k = 0; k < net_tensors->size(); k++)
	{
		if ((*net_tensors)[k].get_name() == top_tensor_name_)
		{
			top_tensor = &((*net_tensors)[k]);
			break;
		}
	}
#ifdef DEBUG
	if (top_tensor == 0)
	{
		cout << "resize_layer.cpp: can't find layer's top tensor!!!!!!!!!!\n" << endl;
		system("pause");
	}
#endif

	int num = bot_tensor->get_shape()[0];
	int channel = bot_tensor->get_shape()[1];
	int height_in = bot_tensor->get_shape()[2];
	int width_in = bot_tensor->get_shape()[3];
	int height_out = output_height_;
	int width_out = output_width_;
	data_type *buffer_in = new data_type[num * height_in * width_in * channel];

	/*copy buffer_in*/
	memcpy(buffer_in, bot_tensor->get_data_ptr()->data_ptr(), num * height_in * width_in * channel * sizeof(data_type));

	int v_step, h_step;
	v_step = height_in * 4096 / height_out;
	h_step = width_in * 4096 / width_out;
	int phase_v, phase_h, inter_phase_v, inter_phase_h;

	if (scaling_method_ == "double")
	{
		for (int n = 0; n < num; n++)
		{
			for (int c = 0; c < channel; c++)
			{
				for (int h = 0; h < height_out; h++)
				{
					for (int w = 0; w < width_out; w++)
					{
						if (h % 2 == 0 && w % 2 == 0)
						{
							top_tensor->set_data(n, c, h, w, buffer_in[((n * channel + c) * height_in + h / 2) * width_in + w / 2]);
						}
						else if (h % 2 == 0 && w % 2 != 0)
						{
							data_type left_val = buffer_in[((n * channel + c) * height_in + h / 2) * width_in + w / 2];
							data_type right_val = buffer_in[((n * channel + c) * height_in + h / 2) * width_in + min(width_in - 1, w / 2 + 1)];
							top_tensor->set_data(n, c, h, w, (left_val + right_val) / 2);
						}
						else if (h % 2 != 0 && w % 2 == 0)
						{
							data_type top_val = buffer_in[((n * channel + c) * height_in + h / 2) * width_in + w / 2];
							data_type bot_val = buffer_in[((n * channel + c) * height_in + min(height_in - 1, h / 2 + 1)) * width_in + w / 2];
							top_tensor->set_data(n, c, h, w, (top_val + bot_val) / 2);
						}
						else
						{
							data_type top_left = buffer_in[((n * channel + c) * height_in + h / 2) * width_in + w / 2];
							data_type top_right = buffer_in[((n * channel + c) * height_in + h / 2) * width_in + min(width_in - 1, w / 2 + 1)];
							data_type bot_left = buffer_in[((n * channel + c) * height_in + min(height_in - 1, h / 2 + 1)) * width_in + w / 2];
							data_type bot_right = buffer_in[((n * channel + c) * height_in + min(height_in - 1, h / 2 + 1)) * width_in + min(width_in - 1, w / 2 + 1)];
							top_tensor->set_data(n, c, h, w, (top_left + top_right + bot_left + bot_right) / 4);
						}
					}
				}
			}
		}
		delete[] buffer_in;
		return;
	}

	/*bilinear, slow*/
	for (int n = 0; n < num; n++)
	{
		for (int c = 0; c < channel; c++)
		{
			phase_v = -(v_step - 4096) / 2;
			for (int h = 0; h < height_out; h++)
			{
				inter_phase_v = ((phase_v - (phase_v / 4096) * 4096) * 32) / 4096;
				phase_h = -(h_step - 4096) / 2;
				for (int w = 0; w < width_out; w++)
				{
					inter_phase_h = ((phase_h - (phase_h / 4096) * 4096) * 32) / 4096;
					int cur_v_pos = max(0, min(height_in - 2, phase_v / 4096));
					int cur_h_pos = max(0, min(width_in - 2, phase_h / 4096));
					if (scaling_method_ == "nearest")
					{
						top_tensor->set_data(n, h, w, c, buffer_in[((n * height_in + cur_v_pos) * width_in + cur_v_pos) * channel + c]);
					}
					else if (scaling_method_ == "bilinear")
					{
						
						/*data_type top_left = buffer_in[((n * height_in + max(0, min(height_in - 1, phase_v / 4096))) * width_in + max(0, min(width_in - 1, phase_h / 4096))) * channel + c];
						data_type top_right = buffer_in[((n * height_in + max(0, min(height_in - 1, phase_v / 4096))) * width_in + max(0, min(width_in - 1, phase_h / 4096 + 1))) * channel + c];
						data_type bot_left = buffer_in[((n * height_in + max(0, min(height_in - 1, phase_v / 4096 + 1))) * width_in + max(0, min(width_in - 1, phase_h / 4096))) * channel + c];
						data_type bot_right = buffer_in[((n * height_in + max(0, min(height_in - 1, phase_v / 4096 + 1))) * width_in + max(0, min(width_in - 1, phase_h / 4096 + 1))) * channel + c];*/
						data_type top_left = buffer_in[((n * height_in + cur_v_pos) * width_in + cur_h_pos) * channel + c];
						data_type top_right = buffer_in[((n * height_in + cur_v_pos) * width_in + cur_h_pos + 1) * channel + c];
						data_type bot_left = buffer_in[((n * height_in + cur_v_pos + 1) * width_in + cur_h_pos) * channel + c];
						data_type bot_right = buffer_in[((n * height_in + cur_v_pos + 1) * width_in + cur_h_pos + 1) * channel + c];
						data_type cur_val = ((32 - inter_phase_v) * (32 - inter_phase_h) * top_left
							+ (32 - inter_phase_v) * inter_phase_h * top_right
							+ inter_phase_v * (32 - inter_phase_h) * bot_left
							+ inter_phase_v * inter_phase_h * bot_right
							) / 32 / 32;
						top_tensor->set_data(n, h, w, c, cur_val);
					}

					phase_h += h_step;
				}
				phase_v += v_step;
			}
		}
	}

	delete[] buffer_in;
}

INSTANTIATE_CLASS(resize_layer_c);