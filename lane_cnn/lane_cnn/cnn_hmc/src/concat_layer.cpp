#include "../include/concat_layer.h"

template <typename data_type>
concat_layer_c<data_type>::concat_layer_c(ifstream &model_file, vector<tensor_c<data_type>> net_tensors)
{
	type_ = "Concat";
	string cur_input;
	model_file >> cur_input;
	name_ = cur_input;
	model_file >> cur_input;
	bot_tensor_names_.push_back(cur_input);
	model_file >> cur_input;
	bot_tensor_names_.push_back(cur_input);
	model_file >> cur_input;
	top_tensor_name_ = cur_input;
	//just support two input now
	input_num_ = 2;

	/*find bot tensors*/
	vector<tensor_c<data_type>*> bot_tensors;
	for (int n = 0; n < input_num_; n++)
	{
		tensor_c<data_type> *cur_bot_tensor = 0;
		for (int k = 0; k < net_tensors.size(); k++)
		{
			if (net_tensors[k].get_name() == bot_tensor_names_[n])
			{
				cur_bot_tensor = &(net_tensors[k]);
				bot_tensors.push_back(cur_bot_tensor);
				break;
			}
		}
#ifdef DEBUG
		if (cur_bot_tensor == 0)
		{
			cout << "concat_layer.cpp: can't find layer's bot tensor!!!!!!!!!!\n" << endl;
			system("pause");
		}
#endif
	}
/*bot tensors should have same dimension*/
#ifdef DEBUG
	int last_num = bot_tensors[0]->get_shape()[0];
	int last_height = bot_tensors[0]->get_shape()[2];
	int last_width = bot_tensors[0]->get_shape()[3];
	for (int k = 1; k < bot_tensors.size(); k++)
	{
		int cur_num = bot_tensors[k]->get_shape()[0];
		int cur_height = bot_tensors[k]->get_shape()[2];
		int cur_width = bot_tensors[k]->get_shape()[3];
		if (cur_num != last_num || cur_height != last_height || cur_width != last_width)
		{
			cout << "concat_layer.cpp: bot tensors don't have same dimension!!!!!!!!!!\n" << endl;
			system("pause");
		}
		last_num = cur_num;
		last_height = cur_height;
		last_width = cur_width;
	}
#endif

	vector<int> bot_shape = bot_tensors[0]->get_shape();
	top_shape_.assign(bot_shape.begin(), bot_shape.end());
	top_shape_[1] = 0;
	for (int k = 0; k < bot_tensors.size(); k++)
	{
		top_shape_[1] += (bot_tensors[k]->get_shape())[1];
	}
}

template <typename data_type>
concat_layer_c<data_type>::concat_layer_c(vector<tensor_c<data_type>*> bot_tensors, string name)
{
	name_ = name;
	input_num_ = bot_tensors.size();
	vector<int> cur_shape, last_shape;
	last_shape = bot_tensors[0]->get_shape();
	int out_channel = last_shape[1];
	bot_tensor_names_.push_back(bot_tensors[0]->get_name());
	for (int n = 1; n < input_num_; n++)
	{
		cur_shape = bot_tensors[n]->get_shape();
#ifdef DEBUG
		if (cur_shape[0] != last_shape[0] || cur_shape[2] != last_shape[2] || cur_shape[3] != last_shape[3])
		{
			cout << "concat_layer.cpp: input tensor's dimension not identical!!!!!!!!!!\n" << endl;
			system("pause");
		}
#endif
		bot_tensor_names_.push_back(bot_tensors[n]->get_name());
		out_channel += cur_shape[1];
		last_shape.clear();
		last_shape.assign(cur_shape.begin(), cur_shape.end());
	}
	top_tensor_name_ = name;
	top_shape_.push_back(last_shape[0]);
	top_shape_.push_back(out_channel);
	top_shape_.push_back(last_shape[2]);
	top_shape_.push_back(last_shape[3]);
}

template <typename data_type>
concat_layer_c<data_type>::~concat_layer_c()
{

}

template <typename data_type>
void concat_layer_c<data_type>::forward(vector<tensor_c<data_type>> *net_tensors)
{
	/*find bot tensors*/
	vector<tensor_c<data_type>*> bot_tensors;
	for (int n = 0; n < input_num_; n++)
	{
		tensor_c<data_type> *cur_bot_tensor = 0;
		for (int k = 0; k < net_tensors->size(); k++)
		{
			if ((*net_tensors)[k].get_name() == bot_tensor_names_[n])
			{
				cur_bot_tensor = &((*net_tensors)[k]);
				bot_tensors.push_back(cur_bot_tensor);
				break;
			}
		}
#ifdef DEBUG
		if (cur_bot_tensor == 0)
		{
			cout << "concat_layer.cpp: can't find layer's bot tensor!!!!!!!!!!\n" << endl;
			system("pause");
		}
#endif
	}

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
		cout << "concat_layer.cpp: can't find layer's top tensor!!!!!!!!!!\n" << endl;
		system("pause");
	}
#endif
	
	int num = top_shape_[0];
	int channel = top_shape_[1];
	int height = top_shape_[2];
	int width = top_shape_[3];
	int channel_index = 0;
	for (int k = 0; k < bot_tensors.size(); k++)
	{
		/*first tensor in bot_tensors is in the front*/
		int cur_channel = bot_tensors[k]->get_shape()[1];
		for (int n = 0; n < num; n++)
		{
			//top_tensor->copy_data(*bot_tensors[k], (n * channel + channel_index) * height * width, n);
			memcpy(top_tensor->get_data_ptr()->data_ptr() + (n * channel + channel_index) * height * width,
				bot_tensors[k]->get_data_ptr()->data_ptr() + n * cur_channel * height * width,
				cur_channel * height * width * sizeof(data_type)
				);
		}
		channel_index += cur_channel;
	}

	//for (int k = 0; k < bot_tensors.size(); k++)
	//{
	//	/*first tensor in bot_tensors is in the front*/
	//	int cur_channel = bot_tensors[k]->get_shape()[1];
	//	for (int n = 0; n < num; n++)
	//	{
	//		for (int h = 0; h < height; h++)
	//		{
	//			for (int w = 0; w < width; w++)
	//			{
	//				for (int c = 0; c < cur_channel; c++)
	//				{
	//					/*dimension is already checked in build function*/
	//					top_tensor->set_data(n, c + channel_index, h, w, bot_tensors[k]->get_data(n, c, h, w));
	//				}
	//			}
	//		}
	//	}
	//	channel_index += cur_channel;
	//}
}

INSTANTIATE_CLASS(concat_layer_c);