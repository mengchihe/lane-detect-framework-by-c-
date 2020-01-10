#include "../include/layer.h"

template <typename data_type>
vector<int> layer_c<data_type>::get_top_shape()
{
	return top_shape_;
}

template <typename data_type>
string layer_c<data_type>::get_layer_name()
{
	return name_;
}

template <typename data_type>
vector<string> layer_c<data_type>::get_bot_tensor_names()
{
	return bot_tensor_names_;
}

template <typename data_type>
string layer_c<data_type>::get_top_tensor_name()
{
	return top_tensor_name_;
}

template <typename data_type>
string layer_c<data_type>::get_type()
{
	return type_;
}

INSTANTIATE_CLASS(layer_c);

