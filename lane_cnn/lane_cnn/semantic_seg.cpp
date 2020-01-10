#include "semantic_seg.h"

template <typename data_type>
semantic_seg_c<data_type>::semantic_seg_c(string input_type)
{
	input_type_ = input_type;
}

template <typename data_type>
void semantic_seg_c<data_type>::semantic_seg_frame(void *input_frame, void *output_frame)
{
	int ori_width = ORI_IMG_WIDTH;
	int ori_height = ORI_IMG_HEIGHT;
	int cnn_in_width = CNN_INPUT_WIDTH;
	int cnn_in_height = CNN_INPUT_HEIGHT;
	byte *ori_buffer = new byte[ori_width * ori_height * 3];
	byte * cnn_in_buffer = new byte[cnn_in_width * cnn_in_height * 3];
	if (input_type_=="mat")
	{
		mat_to_bgr(ori_buffer, (Mat*)input_frame, ori_width, ori_height);
	}		
	//clock_t start, finish;
	//start = clock();
	img_scaling(ori_buffer, cnn_in_buffer, "bilinear");
	//finish = clock();
	//cout << finish - start << ": scaling time\n" << endl;

	vector<tensor_c<data_type>> input_tensors, output_tensors;
	input_tensors.push_back(tensor_c<data_type>(cnn_in_buffer, 1, 3, cnn_in_height, cnn_in_width, "data"));
	input_tensors[0].data_pre_process();
	vector<string> fetch_names;
	//fetch_names.push_back("conv5_3");	//0
	//fetch_names.push_back("score5");	//1
	//fetch_names.push_back("resize4");	//2
	//fetch_names.push_back("score4");	//3
	//fetch_names.push_back("concat4");	//4
	//fetch_names.push_back("resize3");	//5
	//fetch_names.push_back("conv1");	//6
	//fetch_names.push_back("conv2");	//7
	//fetch_names.push_back("conv5_3");	//8
	//fetch_names.push_back("score2");	//9
	//fetch_names.push_back("concat2");	//10
	//fetch_names.push_back("resize1");	//11
	//fetch_names.push_back("score1");	//12
	//fetch_names.push_back("resize1");	//13
	//fetch_names.push_back("concat1");	//14
	//fetch_names.push_back("resize0");	//15
	fetch_names.push_back("output");	//16

	output_tensors = net_.run(input_tensors, fetch_names);

	/*new*/
	data_type aaa = output_tensors[0].get_data(0, 1, 5, 2);

	/*transfer fetch tensors to cnn_out_buffer*/
	int cnn_out_width = output_tensors[0].get_shape()[2];
	int cnn_out_height = output_tensors[0].get_shape()[1];
	byte *cnn_out_buffer = new byte[cnn_in_width * cnn_in_height * 3];
	memcpy(cnn_out_buffer, cnn_in_buffer, cnn_in_width * cnn_in_height * 3 * sizeof(byte));
	draw_lane(cnn_out_buffer, net_.get_tensor("output"));
	//output_tensors[0].to_mem_buff(cnn_out_buffer);

	//FILE *fp_test = fopen("E://1.yuv", "wb");
	//fwrite(cnn_in_buffer, cnn_in_width*cnn_in_height * 3, sizeof(byte), fp_test);
	if (input_type_ == "mat")
		bgr_to_mat((Mat*)output_frame, cnn_out_buffer, cnn_in_width, cnn_in_height);

	delete[] ori_buffer;
	delete[] cnn_in_buffer;
	delete[] cnn_out_buffer;
}

template <typename data_type>
void semantic_seg_c<data_type>::create_net()
{
	net_.create_by_proto("model.txt");
	net_.read_param("weight.txt");

	return;
	//create net in code
	tensor_c<data_type> input_tensor(1, 3, 256, 512, "input");
	net_.add_input_tensor(input_tensor);

	/*kernel_size, out_dims*/
	add_conv_stage("weight_file//encode_conv", "input", 3, 16, "1_1");
	/*kernel_size, stride*/
	add_pooling_layer("relu1_1", 2, 2, "pool1");
	
	add_conv_stage("weight_file//encode_conv", "pool1", 3, 32, "2_1");
	add_pooling_layer("relu2_1", 2, 2, "pool2");

	add_conv_stage("weight_file//encode_conv", "pool2", 3, 64, "3_1");
	add_conv_stage("weight_file//encode_conv", "relu3_1", 1, 32, "3_2");
	add_pooling_layer("relu3_2", 2, 2, "pool3");

	add_conv_stage("weight_file//encode_conv", "pool3", 3, 64, "4_1");
	add_conv_stage("weight_file//encode_conv", "relu4_1", 1, 32, "4_2");
	add_conv_stage("weight_file//encode_conv", "relu4_2", 3, 64, "4_3");
	add_pooling_layer("relu4_3", 2, 2, "pool4");

	add_conv_stage("weight_file//encode_conv", "pool4", 3, 128, "5_1");
	add_conv_stage("weight_file//encode_conv", "relu5_1", 1, 64, "5_2");
	add_conv_stage("weight_file//encode_conv", "relu5_2", 3, 128, "5_3");
	add_pooling_layer("relu5_3", 2, 2, "pool5");

	/*decoder, score5 corresponds to pool5*/
	add_conv_layer("weight_file//decode_score_origin.txt", "pool5", 1, 64, "score5");

	/*deconv4 corresponds the size of pool4*/
	add_resize_layer("score5", net_.get_tensor("score5")->get_shape()[2] * 2,
		net_.get_tensor("score5")->get_shape()[3] * 2, "double", "deconv4");
	/*score1 in tensorflow corresponds to pool4*/
	add_conv_layer("weight_file//decode_score_1.txt", "pool4", 1, 64, "score4");
	add_concat_layer("deconv4", "score4", "concat4");

	add_resize_layer("concat4", net_.get_tensor("concat4")->get_shape()[2] * 2,
		net_.get_tensor("concat4")->get_shape()[3] * 2, "double", "deconv3");
	add_conv_layer("weight_file//decode_score_2.txt", "pool3", 1, 64, "score3");
	add_concat_layer("deconv3", "score3", "concat3");

	add_resize_layer("concat3", net_.get_tensor("concat3")->get_shape()[2] * 2,
		net_.get_tensor("concat3")->get_shape()[3] * 2, "double", "deconv2");
	add_conv_layer("weight_file//decode_score_3.txt", "pool2", 1, 64, "score2");
	add_concat_layer("deconv2", "score2", "concat2");

	add_resize_layer("concat2", net_.get_tensor("concat2")->get_shape()[2] * 2,
		net_.get_tensor("concat2")->get_shape()[3] * 2, "double", "deconv1");
	add_conv_layer("weight_file//decode_score_4.txt", "pool1", 1, 64, "score1");
	add_concat_layer("deconv1", "score1", "concat1");

	/*deconv0 is same size to the input*/
	add_resize_layer("concat1", net_.get_tensor("concat1")->get_shape()[2] * 2,
		net_.get_tensor("concat1")->get_shape()[3] * 2, "double", "deconv0");
	add_conv_layer("weight_file//decode_score_final.txt", "deconv0", 1, 2, "output");
}

template <typename data_type>
void semantic_seg_c<data_type>::add_conv_layer(string param_filename, string bot_tensor_name, int kernel_size, int out_dims, string name)
{
	tensor_c<data_type> *bot_tensor = net_.get_tensor(bot_tensor_name);
	shared_ptr < layer_c<data_type >> cur_layer;
	int stride = 1;
	int bias_en = 0;
	int padding = kernel_size / 2;

	cur_layer.reset(new conv_layer_c<data_type>(param_filename, bot_tensor, bias_en, kernel_size, stride, padding, out_dims, name));
	net_.add_layer(cur_layer);
}

template <typename data_type>
void semantic_seg_c<data_type>::add_resize_layer(string bot_tensor_name, int output_height, int output_width, string scaling_method, string name)
{
	tensor_c<data_type> *bot_tensor = net_.get_tensor(bot_tensor_name);
	shared_ptr < layer_c<data_type >> cur_layer;
	cur_layer.reset(new resize_layer_c<data_type>(bot_tensor, output_height, output_width, scaling_method, name));
	net_.add_layer(cur_layer);
}

template <typename data_type>
void semantic_seg_c<data_type>::add_pooling_layer(string bot_tensor_name, int kernel_size, int stride, string name)
{
	tensor_c<data_type> *bot_tensor = net_.get_tensor(bot_tensor_name);
	shared_ptr < layer_c<data_type >> cur_layer;
	cur_layer.reset(new pooling_layer_c<data_type>("", bot_tensor, kernel_size, stride, name));
	net_.add_layer(cur_layer);
}

template <typename data_type>
void semantic_seg_c<data_type>::add_concat_layer(string bot_tensor_name0, string bot_tensor_name1, string name)
{
	tensor_c<data_type> *bot_tensor0 = net_.get_tensor(bot_tensor_name0);
	tensor_c<data_type> *bot_tensor1 = net_.get_tensor(bot_tensor_name1);
	vector<tensor_c<data_type>*> bot_tensors;
	bot_tensors.push_back(net_.get_tensor(bot_tensor_name0));
	bot_tensors.push_back(net_.get_tensor(bot_tensor_name1));
	shared_ptr < layer_c<data_type >> cur_layer;
	cur_layer.reset(new concat_layer_c<data_type>(bot_tensors, name));
	net_.add_layer(cur_layer);
}

template <typename data_type>
void semantic_seg_c<data_type>::add_conv_stage(string param_path, string bot_tensor_name, int kernel_size, int out_dims, string name)
{
	tensor_c<data_type> *bot_tensor = net_.get_tensor(bot_tensor_name);
	/*
	  use shared_ptr to add each layer
	  if use ordinary ptr, the layer mem will release when this function is end, and the layer_ptr store in net will be invalid
	  if net not store ptr, each layer will force transfer to basic class, derive parts will lost
	*/
	shared_ptr < layer_c<data_type >> cur_layer;
	int stride = 1;
	int bias_en = 0;
	int padding = kernel_size / 2;
	int leaky_en = 0;

	/*bias_en, kernel_size, stride, padding, out_dims*/
	cur_layer.reset(new conv_layer_c<data_type>(param_path + name + "_conv.txt", bot_tensor, bias_en, kernel_size, stride, padding, out_dims, "conv" + name));
	net_.add_layer(cur_layer);

	cur_layer.reset(new batch_norm_layer_c<data_type>(param_path + name + "_bn.txt", net_.get_tensor("conv" + name), "bn" + name));
	net_.add_layer(cur_layer);

	/*leaky_en*/
	cur_layer.reset(new relu_layer_c<data_type>("", net_.get_tensor("bn" + name), leaky_en, "relu" + name));
	net_.add_layer(cur_layer);
}

template <typename data_type>
void semantic_seg_c<data_type>::draw_lane(byte* img_buffer, tensor_c<data_type> *tensor)
{
	int height = tensor->get_shape()[2];
	int width = tensor->get_shape()[3];
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (i == 31 && j == 16)
			{
				int qqq = 3;
			}
			if (tensor->get_data(0, 1, i, j) > tensor->get_data(0, 0, i, j))
			{
				img_buffer[i*width + j] = 128;
				img_buffer[width * height + i*width + j] = 255;
				img_buffer[2 * width * height + i*width + j] = 0;
			}
		}
	}
}

INSTANTIATE_CLASS(semantic_seg_c);