#include "../include/net.h"

template <typename data_type>
void net_c<data_type>::create_by_proto(string proto_filename)
{
	ifstream model_file(proto_filename);
#ifdef DEBUG
	if ((!model_file))
	{
		printf("net.cpp: can't find model file!!!!!!!!!!!!!!!!!\n");
		system("pause");
	}
#endif
	string cur_input;
	shared_ptr < layer_c<data_type >> cur_layer;
	while (model_file.eof()==0)
	{
		model_file >> cur_input;
		/*add current layer into net*/
		if (cur_input == "Data")
			cur_layer.reset(new data_layer_c<data_type>(model_file, tensors_));
		else if (cur_input == "Convolution")
			cur_layer.reset(new conv_layer_c<data_type>(model_file, tensors_));
		else if (cur_input == "BatchNorm")
			cur_layer.reset(new batch_norm_layer_c<data_type>(model_file, tensors_));
		else if (cur_input == "Relu")
			cur_layer.reset(new relu_layer_c<data_type>(model_file, tensors_));
		else if (cur_input == "Pooling")
			cur_layer.reset(new pooling_layer_c<data_type>(model_file, tensors_));
		else if (cur_input == "Concat")
			cur_layer.reset(new concat_layer_c<data_type>(model_file, tensors_));
		else if (cur_input == "Resize")
			cur_layer.reset(new resize_layer_c<data_type>(model_file, tensors_));
		else
		{
			printf("net.cpp: %s, unknown layer type!!!!!!!!!!!!!!!!!\n", cur_input.c_str());
			system("pause");
		}

		/*all layers just need to add layer and top tensors*/
		add_layer(cur_layer);
	}
}

template <typename data_type>
void net_c<data_type>::read_param(string weight_filename)
{
	ifstream weight_file(weight_filename);
#ifdef DEBUG
	if ((!weight_file))
	{
		printf("net.cpp: can't find weight file!!!!!!!!!!!!!!!!!\n");
		system("pause");
	}
#endif
	for (int l = 0; l < layers_.size(); l++)
	{
		layers_[l]->read_param(weight_file);
	}
#ifdef DEBUG
	if (!weight_file.eof())
	{
		printf("net.cpp: parameter file is too long!!!!!!!!!!\n");
		system("pause");
	}
#endif
}

template <typename data_type>
void net_c<data_type>::add_input_tensor(tensor_c<data_type> tensor)
{
#ifdef DEBUG
	map<string, int>::iterator iter = tensors_name_index_.find(tensor.get_name());
	if (iter != tensors_name_index_.end())
	{
		cout << "net.cpp: new tensor's name is already in the net!!!!!!!!!!\n" << endl;
		system("pause");
	}
#endif
	input_tensors_index_.push_back(tensors_.size());
	tensors_name_index_.insert(pair<string, int>(tensor.get_name(), tensors_.size()));
	tensors_.push_back(tensor);
}

template <typename data_type>
void net_c<data_type>::add_tensor(tensor_c<data_type> tensor)
{
#ifdef DEBUG
	map<string, int>::iterator iter = tensors_name_index_.find(tensor.get_name());
	if (iter != tensors_name_index_.end())
	{
		printf("net.cpp: %s, new tensor's name is already in the net!!!!!!!!!!\n", tensor.get_name().c_str());
		system("pause");
	}
#endif
	tensors_name_index_.insert(pair<string, int>(tensor.get_name(), tensors_.size()));
	tensors_.push_back(tensor);
}

template <typename data_type>
void net_c<data_type>::add_layer(shared_ptr<layer_c<data_type>> layer)
{	
	tensor_layer_index_.push_back(layers_.size());
	layers_.push_back(layer);

	tensor_c<data_type> top_tensor(layer->get_top_shape(), layer->get_top_tensor_name());
	add_tensor(top_tensor);
}

template <typename data_type>
tensor_c<data_type> *net_c<data_type>::get_tensor(string name)
{
	map<string, int>::iterator iter = tensors_name_index_.find(name);
#ifdef DEBUG
	if (iter == tensors_name_index_.end())
	{
		printf("net.cpp: %s, can't find tensor in net!!!!!!!!!!\n", name.c_str());
		system("pause");
	}
#endif
	return &tensors_[iter->second];
}

template <typename data_type>
int net_c<data_type>::get_tensor_index(string name)
{
	map<string, int>::iterator iter = tensors_name_index_.find(name);
#ifdef DEBUG
	if (iter == tensors_name_index_.end())
	{
		printf("net.cpp: %s, can't find tensor in net!!!!!!!!!!\n", name.c_str());
		system("pause");
	}
#endif
	return iter->second;
}

template <typename data_type>
vector<tensor_c<data_type>> net_c<data_type>::run(vector<tensor_c<data_type>> input_tensors, vector<string> fetch_names)
{
	int *done_layers = new int[layers_.size()];
	memset(done_layers, 0, layers_.size() * sizeof(int));
	/*copy input tensors into tensors_*/
	for (int k = 0; k < input_tensors.size(); k++)
	{
		int cur_index = get_tensor_index(input_tensors[k].get_name());		
		tensors_[cur_index].copy_from(input_tensors[k]);
		/*input layer is done*/
		done_layers[tensor_layer_index_[cur_index]] = 1;
	}
	
	/*find layers should be executed backward from fetch tensors*/
	for (int k = 0; k < fetch_names.size(); k++)
	{
		/*find current tensors in the net*/
		int tensor_index = get_tensor_index(fetch_names[k]);
		tensor_done_judge(tensor_index, done_layers);
	}

	/*run layers according to done_layers from front to rear*/
	for (int l = 0; l < layers_.size(); l++)
	{
		if (done_layers[l] == 0)
			continue;
		clock_t start, finish;
		start = clock();

		layers_[l]->forward(&tensors_);

		finish = clock();
		cout << finish - start<<": " << layers_[l]->get_layer_name() << endl;
	}
	
	/*add fetch tensor into output vector*/
	vector<tensor_c<data_type>> fetch_tensors;
	for (int k = 0; k < fetch_names.size(); k++)
	{
		map<string, int>::iterator fetch_iter = tensors_name_index_.find(fetch_names[k]);
#ifdef DEBUG
		if (fetch_iter == tensors_name_index_.end())
		{
			cout << "net.cpp: can't find fetch tensor in net!!!!!!!!!!\n" << endl;
			system("pause");
		}
#endif
		fetch_tensors.push_back(tensors_[fetch_iter->second]);
	}

	delete[] done_layers;
	return fetch_tensors;
}

template <typename data_type>
void net_c<data_type>::tensor_done_judge(int tensor_index, int *done_layers)
{
	int layer_index = tensor_layer_index_[tensor_index];
	if (done_layers[layer_index] == 0)
	{
		done_layers[layer_index] = 1;
		for (int k = 0; k < layers_[layer_index]->get_bot_tensor_names().size(); k++)
		{
			int tensor_index_new = get_tensor_index(layers_[layer_index]->get_bot_tensor_names()[k]);
			tensor_done_judge(tensor_index_new, done_layers);
		}
	}
	else
	{
		return;
	}
}

INSTANTIATE_CLASS(net_c);