#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include "cnn_common.h"
#include "layer.h"

template <typename data_type>
class conv_layer_c: public layer_c<data_type>
{
public:
	conv_layer_c() {};
	explicit conv_layer_c(ifstream &model_file, vector<tensor_c<data_type>> net_tensors);
	explicit conv_layer_c(string parameter_file, tensor_c<data_type> *bot_tensor,
		int bias_en, int kernel_size, int stride, int padding, int out_dims, string name);
	~conv_layer_c();

	virtual void read_param(ifstream &weight_file);
	virtual void forward(vector<tensor_c<data_type>> *net_tensors);
	void matrix_multiply(data_type *mat_in0, data_type *mat_in1, data_type *mat_out, int dim0, int dim1, int dim2);

protected:
	int bias_en_;
	int kernel_size_;
	int stride_;
	int padding_;
	int in_dims_;
	int out_dims_;
};

#endif