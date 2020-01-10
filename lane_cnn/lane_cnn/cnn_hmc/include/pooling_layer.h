#ifndef POOLING_LAYER_H
#define POOLING_LAYER_H

#include "cnn_common.h"
#include "layer.h"

template <typename data_type>
class pooling_layer_c : public layer_c<data_type>
{
public:
	pooling_layer_c() {};
	explicit pooling_layer_c(ifstream &model_file, vector<tensor_c<data_type>> net_tensors);
	explicit pooling_layer_c(string parameter_file, tensor_c<data_type> *bot_tensor,
		int kernel_size, int stride, string name);
	~pooling_layer_c();

	virtual void forward(vector<tensor_c<data_type>> *net_tensors);

protected:
	int kernel_size_;
	int stride_;
	string pooling_method_;
};

#endif