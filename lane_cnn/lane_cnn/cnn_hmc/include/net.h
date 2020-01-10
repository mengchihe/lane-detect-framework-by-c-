#ifndef NET_H
#define NET_H

#include "cnn_common.h"
#include "layer.h"
#include "data_layer.h"
#include "conv_layer.h"
#include "batch_norm_layer.h"
#include "concat_layer.h"
#include "pooling_layer.h"
#include "relu_layer.h"
#include "resize_layer.h"

template <typename data_type>
class net_c
{
public:
	net_c() 
	{
		support_layer_names_.push_back("Data");
		support_layer_names_.push_back("Convolution");
		support_layer_names_.push_back("BatchNorm");
		support_layer_names_.push_back("Relu");
		support_layer_names_.push_back("Pooling");
		support_layer_names_.push_back("Concat");
		support_layer_names_.push_back("Resize");
	};

	void create_by_proto(string proto_filename);

	void read_param(string weight_filename);

	void add_input_tensor(tensor_c<data_type> tensor);

	/*add a tensor into the net, which is always between two layers or output*/
	void add_tensor(tensor_c<data_type> tensor);

	/*add current layer and its output tensor into the net*/
	void add_layer(shared_ptr<layer_c<data_type>> layer);

	tensor_c<data_type> *get_tensor(string name);

	int get_tensor_index(string name);

	vector<tensor_c<data_type>> run(vector<tensor_c<data_type>> input_tensors, vector<string> fetch_names);

	/*
	  recurrent function, whether the layer outputs current tensor is done
	  if done return, otherwise execute function of previous tensor index
	*/
	void tensor_done_judge(int tensor_index, int *done_layers);
protected:
	/*all support layers*/
	vector<string> support_layer_names_;

	/*
	  store shared_ptr for each layer in order
	  if use ordinary ptr, the layer mem will release when create function is end, and the layer_ptr store in net will be invalid
	  if net not store ptr, each layer will force transfer to basic class, derive part will lost
	*/
	vector<shared_ptr<layer_c<data_type>>> layers_;

	/*data between layers*/
	vector<tensor_c<data_type>> tensors_;

	/*each tensor is which layer's top tensor*/
	vector<int> tensor_layer_index_;

	/*tensor name and index in tensors_*/
	map<string, int> tensors_name_index_;

	/*position of input tensors in tensors_, net_c.run() will copy input tensor into tensors_ by this index*/
	vector<int> input_tensors_index_;

};

#endif