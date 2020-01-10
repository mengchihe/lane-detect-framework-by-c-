#ifndef RESIZE_LAYER_H
#define RESIZE_LAYER_H

#include "cnn_common.h"
#include "layer.h"

template <typename data_type>
class resize_layer_c :public layer_c<data_type>
{
public:
	resize_layer_c() {};
	explicit resize_layer_c(ifstream &model_file, vector<tensor_c<data_type>> net_tensors);
	explicit resize_layer_c(tensor_c<data_type> *bot_tensor, int output_height, int output_width, string scaling_method, string name);
	~resize_layer_c();

	virtual void forward(vector<tensor_c<data_type>> *net_tensors);

protected:
	int output_height_;
	int output_width_;
	double h_ratio_;
	double w_ratio_;
	string scaling_method_;
};

#endif