#include "../include/relu_layer.h"


template <typename data_type>
relu_layer_c<data_type>::relu_layer_c(ifstream &model_file, vector<tensor_c<data_type>> net_tensors)
{
	type_ = "Relu";
	string cur_input;
	model_file >> cur_input;
	name_ = cur_input;
	model_file >> cur_input;
	bot_tensor_names_.push_back(cur_input);
	model_file >> cur_input;
	top_tensor_name_ = cur_input;
	leaky_en_ = 0;

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
		cout << "relu_layer.cpp: can't find layer's bot tensor!!!!!!!!!!\n" << endl;
		system("pause");
	}
#endif
	vector<int> bot_shape = bot_tensor->get_shape();
	top_shape_.assign(bot_shape.begin(), bot_shape.end());
}

template <typename data_type>
relu_layer_c<data_type>::relu_layer_c(string parameter_file, tensor_c<data_type> *bot_tensor, int leaky_en, string name)
{
	name_ = name;
	bot_tensor_names_.push_back(bot_tensor->get_name());
	top_tensor_name_ = name;
	vector<int> bot_shape = bot_tensor->get_shape();
	top_shape_.push_back(bot_shape[0]);
	top_shape_.push_back(bot_shape[1]);
	top_shape_.push_back(bot_shape[2]);
	top_shape_.push_back(bot_shape[3]);

	leaky_en_ = leaky_en;
	if (leaky_en == 0)
		return;
}

template <typename data_type>
relu_layer_c<data_type>::~relu_layer_c()
{

}

template <typename data_type>
void relu_layer_c<data_type>::forward(vector<tensor_c<data_type>> *net_tensors)
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
		cout << "relu_layer.cpp: can't find layer's bot tensor!!!!!!!!!!\n" << endl;
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
		cout << "relu_layer.cpp: can't find layer's top tensor!!!!!!!!!!\n" << endl;
		system("pause");
	}
#endif
	int num = bot_tensor->get_shape()[0];
	int channel = bot_tensor->get_shape()[1];
	int height = bot_tensor->get_shape()[2];
	int width = bot_tensor->get_shape()[3];

	for (int n = 0; n < num; n++)
	{
		for (int c = 0; c < channel; c++)
		{
			for (int h = 0; h < height; h++)
			{
				for (int w = 0; w < width; w++)
				{
					data_type cur_val = bot_tensor->get_data(n, c, h, w);
					if (leaky_en_ == 0)
					{
						cur_val = max(0, cur_val);
					}
					else
					{

					}
					top_tensor->set_data(n, c, h, w, cur_val);
				}
			}
		}
	}
}

INSTANTIATE_CLASS(relu_layer_c);