#ifndef LANE_DETECT_H
#define LANE_DETECT_H

#include "common.h"
#include "cnn_hmc\include\cnn_hmc.h"

template <typename data_type>
class semantic_seg_c
{
public:
	semantic_seg_c() {};
	semantic_seg_c(string input_type);

	void semantic_seg_frame(void *input_frame, void *output_frame);
	void create_net();
	void add_conv_layer(string param_filename, string bot_tensor_name, int kernel_size, int out_dims, string name);
	void add_resize_layer(string bot_tensor_name, int output_height, int output_width, string scaling_method, string name);
	void add_pooling_layer(string bot_tensor_name, int kernel_size, int stride, string name);
	void add_concat_layer(string bot_tensor_name0, string bot_tensor_name1, string name);
	/*add connected conv, bn and relu layer, height and width is not change*/
	void add_conv_stage(string param_path, string bot_tensor_name, int kernel_size, int out_dims, string name);

	void draw_lane(byte* img_buffer, tensor_c<data_type> *tensor);
protected:
	string input_type_;
	net_c<data_type> net_;
};


#endif
