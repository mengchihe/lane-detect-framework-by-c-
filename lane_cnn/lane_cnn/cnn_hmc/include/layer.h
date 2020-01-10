#ifndef LAYER_H
#define LAYER_H

#include "cnn_common.h"
#include "tensor.h"

template <typename data_type>
class net_c;

template <typename data_type>
class layer_c
{
public:
	layer_c() {};

	/*
	  each layer should able to get tensors in the net
	  must use pointer as parameter, otherwise function will copy a new vector<tensor_c> in it,
	  and this will cause error when tensor_c's constructor doesn't copy the whole class completely
	*/
	virtual void read_param(ifstream &weight_file) {};
	virtual void forward(vector<tensor_c<data_type>> *net_tensors) {};
	virtual void backward() {};
	virtual void read_param() {};
	vector<int> get_top_shape();
	string get_layer_name();
	vector<string> get_bot_tensor_names();
	string get_top_tensor_name();
	string get_type();

protected:
	vector<tensor_c<data_type>> parameters_;
	vector<string> bot_tensor_names_;
	/*top tensor name always identical to layer name*/
	string top_tensor_name_;
	vector<int> top_shape_;
	string name_;
	string type_;
};

#endif