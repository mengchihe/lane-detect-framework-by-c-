#ifndef BATCH_NORM_LAYER
#define BATCH_NORM_LAYER

#include "cnn_common.h"
#include "layer.h"

template <typename data_type>
class batch_norm_layer_c : public layer_c<data_type>
{
public:
	batch_norm_layer_c() {};
	explicit batch_norm_layer_c(ifstream &model_file, vector<tensor_c<data_type>> net_tensors);
	explicit batch_norm_layer_c(string parameter_file, tensor_c<data_type> *bot_tensor, string name);
	~batch_norm_layer_c();

	virtual void read_param(ifstream &weight_file);
	virtual void forward(vector<tensor_c<data_type>> *net_tensors);

protected:
	data_type moving_mean_;
	data_type moving_variance_;
	data_type gamma_;
	data_type beta_;
	data_type const_;
	//string name_;
};

#endif