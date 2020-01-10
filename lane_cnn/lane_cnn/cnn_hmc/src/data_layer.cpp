#include "../include/data_layer.h"

template <typename data_type>
data_layer_c<data_type>::data_layer_c(ifstream &model_file, vector<tensor_c<data_type>> net_tensors)
{
	type_ = "Data";
	string cur_input;
	model_file >> cur_input;
	name_ = cur_input;
	model_file >> cur_input;
	top_tensor_name_ = cur_input;
	int param_read_success[4] = { 0 };
	for (int k = 0; k < 4; k++)
	{
		model_file >> cur_input;
		if (strncmp(cur_input.c_str(), "n=", 2)==0)
		{
			num_ = atoi(cur_input.substr(2, cur_input.length()-1).c_str());
			param_read_success[0] = 1;
		}
		else if (strncmp(cur_input.c_str(), "c=", 2) == 0)
		{
			channel_ = atoi(cur_input.substr(2, cur_input.length() - 1).c_str());
			param_read_success[1] = 1;
		}
		else if (strncmp(cur_input.c_str(), "h=", 2) == 0)
		{
			height_ = atoi(cur_input.substr(2, cur_input.length() - 1).c_str());
			param_read_success[2] = 1;
		}
		else if (strncmp(cur_input.c_str(), "w=", 2) == 0)
		{
			width_ = atoi(cur_input.substr(2, cur_input.length() - 1).c_str());
			param_read_success[3] = 1;
		}
		else
		{
			printf("data_layer.cpp: layer = %s, param = %s, unknown parameter!!!!!!!!!!!!!!!!!\n", name_.c_str(), cur_input.c_str());
			system("pause");
		}
	}
#ifdef DEBUG
	if (param_read_success[0] == 0)
	{
		printf("data_layer.cpp: layer = %s, param = n, undefined parameter!!!!!!!!!!!!!!!!!\n", name_.c_str());
		system("pause");
	}
	if (param_read_success[1] == 0)
	{
		printf("data_layer.cpp: layer = %s, param = c, undefined parameter!!!!!!!!!!!!!!!!!\n", name_.c_str());
		system("pause");
	}
	if (param_read_success[2] == 0)
	{
		printf("data_layer.cpp: layer = %s, param = h, undefined parameter!!!!!!!!!!!!!!!!!\n", name_.c_str());
		system("pause");
	}
	if (param_read_success[3] == 0)
	{
		printf("data_layer.cpp: layer = %s, param = w, undefined parameter!!!!!!!!!!!!!!!!!\n", name_.c_str());
		system("pause");
	}
#endif
	top_shape_.push_back(num_);
	top_shape_.push_back(channel_);
	top_shape_.push_back(height_);
	top_shape_.push_back(width_);
}

template <typename data_type>
data_layer_c<data_type>::data_layer_c(int num, int channel, int height, int width)
{

}

template <typename data_type>
data_layer_c<data_type>::~data_layer_c()
{

}

INSTANTIATE_CLASS(data_layer_c);