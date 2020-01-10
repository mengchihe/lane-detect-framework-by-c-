#include "../include/pooling_layer.h"

template <typename data_type>
pooling_layer_c<data_type>::pooling_layer_c(ifstream &model_file, vector<tensor_c<data_type>> net_tensors)
{
	type_ = "Pooling";
	string cur_input;
	model_file >> cur_input;
	name_ = cur_input;
	model_file >> cur_input;
	bot_tensor_names_.push_back(cur_input);
	model_file >> cur_input;
	top_tensor_name_ = cur_input;
	pooling_method_ = "max";

	int param_read_success[2] = { 0 };
	for (int k = 0; k < 2; k++)
	{
		model_file >> cur_input;
		if (strncmp(cur_input.c_str(), "k=", 2) == 0)
		{
			kernel_size_ = atoi(cur_input.substr(2, cur_input.length() - 1).c_str());
			param_read_success[0] = 1;
		}
		else if (strncmp(cur_input.c_str(), "s=", 2) == 0)
		{
			stride_ = atoi(cur_input.substr(2, cur_input.length() - 1).c_str());
			param_read_success[1] = 1;
		}
		else
		{
			printf("pooling_layer.cpp: layer = %s, param = %s, unknown parameter!!!!!!!!!!!!!!!!!\n", name_.c_str(), cur_input.c_str());
			system("pause");
		}
	}
#ifdef DEBUG
	if (param_read_success[0] == 0)
	{
		printf("pooling_layer.cpp: layer = %s, param = k, undefined parameter!!!!!!!!!!!!!!!!!\n", name_.c_str());
		system("pause");
	}
	if (param_read_success[1] == 0)
	{
		printf("pooling_layer.cpp: layer = %s, param = c, undefined parameter!!!!!!!!!!!!!!!!!\n", name_.c_str());
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
		cout << "pooling_layer.cpp: can't find layer's bot tensor!!!!!!!!!!\n" << endl;
		system("pause");
	}
#endif
	vector<int> bot_shape = bot_tensor->get_shape();
	int in_h = bot_shape[2];
	int in_w = bot_shape[3];
	top_shape_.push_back(bot_shape[0]);
	top_shape_.push_back(bot_shape[1]);
	top_shape_.push_back(in_h / stride_);
	top_shape_.push_back(in_w / stride_);
}

template <typename data_type>
pooling_layer_c<data_type>::pooling_layer_c(string parameter_file, tensor_c<data_type> *bot_tensor,
	int kernel_size, int stride, string name)
{
	name_ = name;
	pooling_method_ = "max";
	kernel_size_ = kernel_size;
	stride_ = stride;
	bot_tensor_names_.push_back(bot_tensor->get_name());
	top_tensor_name_ = name;

	vector<int> bot_shape = bot_tensor->get_shape();
	int in_h = bot_shape[2];
	int in_w = bot_shape[3];
	top_shape_.push_back(bot_shape[0]);
	top_shape_.push_back(bot_shape[1]);
	top_shape_.push_back(in_h / stride);
	top_shape_.push_back(in_w / stride);
}

template <typename data_type>
pooling_layer_c<data_type>::~pooling_layer_c()
{

}

template <typename data_type>
void pooling_layer_c<data_type>::forward(vector<tensor_c<data_type>> *net_tensors)
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
		cout << "pooling_layer.cpp: can't find layer's bot tensor!!!!!!!!!!\n" << endl;
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
		cout << "pooling_layer.cpp: can't find layer's top tensor!!!!!!!!!!\n" << endl;
		system("pause");
	}
#endif
	int num = top_shape_[0];
	int channel = top_shape_[1];
	int h_out = top_shape_[2];
	int w_out = top_shape_[3];

	for (int n = 0; n < num; n++)
	{
		for (int c = 0; c < channel; c++)
		{
			for (int h = 0; h < h_out; h++)
			{
				for (int w = 0; w < w_out; w++)
				{
					data_type out_val = bot_tensor->get_data(n, c, h * stride_, w * stride_);
					if (pooling_method_ == "max")
					{
						for (int i = 0; i < kernel_size_; i++)
						{
							for (int j = 0; j < kernel_size_; j++)
							{
								data_type iter_val = bot_tensor->get_data(n, c, h * stride_ + i, w * stride_ + j);
								if (iter_val > out_val)
								{
									out_val = iter_val;
								}
							}
						}
					}
					top_tensor->set_data(n, c, h, w, out_val);
				}
			}
		}
	}
}

INSTANTIATE_CLASS(pooling_layer_c);