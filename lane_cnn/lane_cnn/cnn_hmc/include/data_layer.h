#ifndef DATA_LAYER_H
#define DATA_LAYER_H

#include "cnn_common.h"
#include "layer.h"

template <typename data_type>
class data_layer_c : public layer_c<data_type>
{
public:
	data_layer_c() {};
	explicit data_layer_c(ifstream &model_file, vector<tensor_c<data_type>> net_tensors);
	explicit data_layer_c(int num, int channel, int height, int width);
	~data_layer_c();

protected:
	int num_;
	int channel_;
	int height_;
	int width_;
};

#endif