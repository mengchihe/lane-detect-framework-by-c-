#ifndef CONCAT_LAYER_H
#define CONCAT_LAYER_H

#include "cnn_common.h"
#include "layer.h"

template <typename data_type>
class concat_layer_c : public layer_c<data_type>
{
public:
	concat_layer_c() {};
	explicit concat_layer_c(ifstream &model_file, vector<tensor_c<data_type>> net_tensors);
	explicit concat_layer_c(vector<tensor_c<data_type>*> bot_tensors, string name);
	~concat_layer_c();

	virtual void forward(vector<tensor_c<data_type>> *net_tensors);
protected:
	int input_num_;
};

#endif