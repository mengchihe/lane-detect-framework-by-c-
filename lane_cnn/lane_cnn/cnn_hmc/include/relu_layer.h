#ifndef RELU_LAYER_H
#define RELU_LAYER_H

#include "cnn_common.h"
#include "layer.h"

template <typename data_type>
class relu_layer_c : public layer_c<data_type>
{
public:
	relu_layer_c() {};
	explicit relu_layer_c(ifstream &model_file, vector<tensor_c<data_type>> net_tensors);
	explicit relu_layer_c(string parameter_file, tensor_c<data_type> *bot_tensor, int leaky_en, string name);
	~relu_layer_c();

	virtual void forward(vector<tensor_c<data_type>> *net_tensors);

protected:
	int leaky_en_;
	data_type leak_param_;
};

#endif