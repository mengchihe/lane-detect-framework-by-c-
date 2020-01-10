#include "../include/conv_layer.h"

template <typename data_type>
conv_layer_c<data_type>::conv_layer_c(ifstream &model_file, vector<tensor_c<data_type>> net_tensors)
{
	type_ = "Convolution";
	string cur_input;
	model_file >> cur_input;
	name_ = cur_input;
	model_file >> cur_input;
	bot_tensor_names_.push_back(cur_input);
	model_file >> cur_input;
	top_tensor_name_ = cur_input;

	bias_en_ = 0;
	int param_read_success[4] = { 0 };
	for (int k = 0; k < 4; k++)
	{
		model_file >> cur_input;
		if (strncmp(cur_input.c_str(), "k=", 2) == 0)
		{
			kernel_size_ = atoi(cur_input.substr(2, cur_input.length() - 1).c_str());
			param_read_success[0] = 1;
		}
		else if (strncmp(cur_input.c_str(), "c=", 2) == 0)
		{
			out_dims_ = atoi(cur_input.substr(2, cur_input.length() - 1).c_str());
			param_read_success[1] = 1;
		}
		else if (strncmp(cur_input.c_str(), "s=", 2) == 0)
		{
			stride_ = atoi(cur_input.substr(2, cur_input.length() - 1).c_str());
			param_read_success[2] = 1;
		}
		else if (strncmp(cur_input.c_str(), "p=", 2) == 0)
		{
			padding_ = atoi(cur_input.substr(2, cur_input.length() - 1).c_str());
			param_read_success[3] = 1;
		}
		else
		{
			printf("conv_layer.cpp: layer = %s, param = %s, unknown parameter!!!!!!!!!!!!!!!!!\n", name_.c_str(), cur_input.c_str());
			system("pause");
		}
	}
#ifdef DEBUG
	if (param_read_success[0] == 0)
	{
		printf("conv_layer.cpp: layer = %s, param = k, undefined parameter!!!!!!!!!!!!!!!!!\n", name_.c_str());
		system("pause");
	}
	if (param_read_success[1] == 0)
	{
		printf("conv_layer.cpp: layer = %s, param = c, undefined parameter!!!!!!!!!!!!!!!!!\n", name_.c_str());
		system("pause");
	}
	if (param_read_success[2] == 0)
	{
		printf("conv_layer.cpp: layer = %s, param = s, undefined parameter!!!!!!!!!!!!!!!!!\n", name_.c_str());
		system("pause");
	}
	if (param_read_success[3] == 0)
	{
		printf("conv_layer.cpp: layer = %s, param = p, undefined parameter!!!!!!!!!!!!!!!!!\n", name_.c_str());
		system("pause");
	}
#endif	

	/*find bot tensor*/
	tensor_c<data_type> *bot_tensor = 0;
	for (int k = 0; k < net_tensors.size(); k++)
	{
		if (net_tensors[k].get_name() == bot_tensor_names_[0])
		{
			bot_tensor = &(net_tensors[k]);
			break;
		}
	}
#ifdef DEBUG
	if (bot_tensor == 0)
	{
		cout << "conv_layer.cpp: can't find layer's bot tensor!!!!!!!!!!\n" << endl;
		system("pause");
	}
#endif
	vector<int> bot_shape = bot_tensor->get_shape();
	int in_dims = bot_shape[1];
	int in_h = bot_shape[2];
	int in_w = bot_shape[3];
	int out_h = (in_h - kernel_size_ + 2 * padding_) / stride_ + 1;
	int out_w = (in_w - kernel_size_ + 2 * padding_) / stride_ + 1;
	top_shape_.push_back(bot_shape[0]);
	top_shape_.push_back(out_dims_);
	top_shape_.push_back(out_h);
	top_shape_.push_back(out_w);
	in_dims_ = in_dims;
}

/*
  bot_tensor should use pointer, otherwise will create a formal parameter,
  this formal parameter will use default construct function which will not malloc memory,
  and will cause error when release this formal parameter at end of the function
*/
template <typename data_type>
conv_layer_c<data_type>::conv_layer_c(string parameter_file, tensor_c<data_type> *bot_tensor,
	int bias_en, int kernel_size, int stride, int padding, int out_dims, string name)
{
	bias_en_ = bias_en;
	kernel_size_ = kernel_size;
	stride_ = stride;
	padding_ = padding;
	out_dims_ = out_dims;
	name_ = name;
	bot_tensor_names_.push_back(bot_tensor->get_name());
	top_tensor_name_ = name;
	
	vector<int> bot_shape = bot_tensor->get_shape();
	int in_dims = bot_shape[1];
	int in_h = bot_shape[2];
	int in_w = bot_shape[3];
	int out_h = (in_h - kernel_size + 2 * padding) / stride + 1;
	int out_w = (in_w - kernel_size + 2 * padding) / stride + 1;
	top_shape_.push_back(bot_shape[0]);
	top_shape_.push_back(out_dims);
	top_shape_.push_back(out_h);
	top_shape_.push_back(out_w);

	/*
	  read parameter from input file
	  parameter dimension: kernel_size * kernel_size * channel_in * channel_out
	*/
	tensor_c<data_type> conv_w(kernel_size, kernel_size, in_dims, out_dims);
	int data_len = kernel_size*kernel_size*in_dims*out_dims;
	ifstream weight_file(parameter_file);
#ifdef DEBUG
	if (!weight_file)
	{
		printf("conv_layer.cpp: can't find parameter file!!!!!!!!!!!!!!!!!\n");
		system("pause");
	}
#endif
	for (int i = 0; i < data_len; i++)
	{
#ifdef DEBUG
		if (weight_file.eof())
		{
			printf("conv_layer.cpp: parameter file is not long enough!!!!!!!!!!\n");
			system("pause");
		}
#endif
		data_type cur_val;
		weight_file >> cur_val;
		conv_w.set_data(i, cur_val);
	}
#ifdef DEBUG
	if (!weight_file.eof())
	{
		printf("conv_layer.cpp: parameter file is too long!!!!!!!!!!\n");
		system("pause");
	}
#endif
	parameters_.push_back(conv_w);
	
}

template <typename data_type>
conv_layer_c<data_type>::~conv_layer_c()
{

}

template <typename data_type>
void conv_layer_c<data_type>::read_param(ifstream &weight_file)
{
	tensor_c<data_type> conv_w(kernel_size_, kernel_size_, in_dims_, out_dims_);
	int data_len = kernel_size_ * kernel_size_ * in_dims_ * out_dims_;
	for (int i = 0; i < data_len; i++)
	{
#ifdef DEBUG
		if (weight_file.eof())
		{
			printf("conv_layer.cpp: parameter file is not long enough!!!!!!!!!!\n");
			system("pause");
		}
#endif
		data_type cur_val;
		weight_file >> cur_val;
		conv_w.set_data(i, cur_val);
	}
	parameters_.push_back(conv_w);
}

template <typename data_type>
void conv_layer_c<data_type>::forward(vector<tensor_c<data_type>> *net_tensors)
{
	clock_t start, finish;
	start = clock();
	/*find bot tensor*/
	tensor_c<data_type> *bot_tensor=0;
	for (int k = 0; k < net_tensors->size(); k++)
	{
		if ((*net_tensors)[k].get_name() == bot_tensor_names_[0])
		{
			bot_tensor = &((*net_tensors)[k]);
			break;
		}
	}
#ifdef DEBUG
	if (bot_tensor == 0)
	{
		cout << "conv_layer.cpp: can't find layer's bot tensor!!!!!!!!!!\n" << endl;
		system("pause");
	}
#endif
	/*find top tensor*/
	tensor_c<data_type> *top_tensor=0;
	for (int k = 0; k < net_tensors->size(); k++)
	{
		if ((*net_tensors)[k].get_name() == top_tensor_name_)
		{
			top_tensor = &((*net_tensors)[k]);
			break;
		}
	}
#ifdef DEBUG
	if (top_tensor == 0)
	{
		cout << "conv_layer.cpp: can't find layer's top tensor!!!!!!!!!!\n" << endl;
		system("pause");
	}
#endif
	
	int num = bot_tensor->get_shape()[0];
	int channel_in = bot_tensor->get_shape()[1];
	int height_in = bot_tensor->get_shape()[2];
	int width_in = bot_tensor->get_shape()[3];
	int height_out = (height_in - kernel_size_ + 2 * padding_) / stride_ + 1;
	int width_out = (width_in - kernel_size_ + 2 * padding_) / stride_ + 1;
	int channel_out = out_dims_;
	if(num != top_tensor->get_shape()[0] 
		|| channel_out != top_tensor->get_shape()[1] 
		|| height_out != top_tensor->get_shape()[2] 
		|| width_out != top_tensor->get_shape()[3])
		top_tensor->reshape(num, channel_out, height_out, width_out);
	/*read into temp buffer to speed up*/
	data_type *buffer_in = new data_type[num * (height_in + 2 * padding_) * (width_in + 2 * padding_) * channel_in];
	data_type *buffer_weight = new data_type[kernel_size_ * kernel_size_ * channel_in * channel_out];
	//memset(buffer_in, 0, num * (height_in + 2 * padding_) * (width_in + 2 * padding_) * channel_in*sizeof(data_type));
	//memset(buffer_weight, 0, kernel_size_ * kernel_size_ * channel_in * channel_out*sizeof(data_type));

	finish = clock();
	//cout << finish - start << ": input" << endl;

	start = clock();
	/*copy buffer_in*/
	int height_pad = height_in + 2 * padding_;
	int width_pad = width_in + 2 * padding_;
	for (int n = 0; n < num; n++)
	{
		for (int c = 0; c < channel_in; c++)
		{
			/*copy one line, can't memcpy whole frame because need padding on border*/
			for (int h = padding_; h < height_in + padding_; h++)
			{
				memcpy(buffer_in + ((n * channel_in + c) * height_pad + h) * width_pad + padding_,
					bot_tensor->get_data_ptr()->data_ptr() + ((n * channel_in + c) * height_in + h - padding_) * width_in,
					width_in * sizeof(data_type));
				/*supply left part*/
				for (int w = 0; w < padding_; w++)
				{
					buffer_in[((n*channel_in + c) * height_pad + h) * width_pad + w]
						= buffer_in[((n*channel_in + c) * height_pad + h) * width_pad + padding_];
				}
				/*supply right part*/
				for (int w = width_in + padding_; w < width_pad; w++)
				{
					buffer_in[((n*channel_in + c) * height_pad + h) * width_pad + w]
						= buffer_in[((n*channel_in + c) * height_pad + h) * width_pad + width_in + padding_ - 1];
				}
			}
			/*supply top part*/
			for (int h = 0; h < padding_; h++)
			{
				memcpy(buffer_in + ((n * channel_in + c) * height_pad + h) * width_pad,
					buffer_in + ((n * channel_in + c) * height_pad + padding_) * width_pad,
					width_pad * sizeof(data_type));
			}
			/*supply bot part*/
			for (int h = height_in + padding_; h < height_pad; h++)
			{
				memcpy(buffer_in + ((n * channel_in + c) * height_pad + h) * width_pad,
					buffer_in + ((n * channel_in + c) * height_pad + height_in + padding_ - 1) * width_pad,
					width_pad * sizeof(data_type));
			}
		}
	}

	//height_in += 2 * padding_;
	//width_in += 2 * padding_;
	//int height_pad = height_in + 2 * padding_;
	//int width_pad = width_in + 2 * padding_;
	/*copy buffer_weight*/
	memcpy(buffer_weight, parameters_[0].get_data_ptr()->data_ptr(), kernel_size_ * kernel_size_ * channel_in * channel_out * sizeof(data_type));

	finish = clock();
	//cout << finish - start << ": read input" << endl;

	start = clock();
	/*transfer input and weight into matrix*/
	data_type *data_matrix = new data_type[kernel_size_ * kernel_size_ * channel_in * height_out * width_out];
	memset(data_matrix, 0, kernel_size_ * kernel_size_ * channel_in * height_out * width_out * sizeof(data_type));
	data_type *weight_matrix = new data_type[channel_out * kernel_size_ * kernel_size_ * channel_in];
	memset(weight_matrix, 0, channel_out * kernel_size_ * kernel_size_ * channel_in * sizeof(data_type));
	/*transfer weight matrix*/
	for (int c_out = 0; c_out < channel_out; c_out++)
	{
		for (int k0 = 0; k0 < kernel_size_; k0++)
		{
			for (int k1 = 0; k1 < kernel_size_; k1++)
			{
				for (int c_in = 0; c_in < channel_in; c_in++)
				{
					/*
					  ori dimension: k0, k1, c_in, c_out
					  matrix dimension: c_out, c_in, k0, k1, which equals to c_out, (c_in*k0*k1)
					*/
					weight_matrix[((c_out * channel_in + c_in) * kernel_size_ + k0) * kernel_size_ + k1]
						= buffer_weight[((k0 * kernel_size_ + k1) * channel_in + c_in) * channel_out + c_out];
				}
			}
		}
	}

	/*transfer data matrix*/
	/*memcpy method, fast, but stride must be 1*/
	if(stride_ == 1)
	{
		for (int c_in = 0; c_in < channel_in; c_in++)
		{
			for (int k0 = 0; k0 < kernel_size_; k0++)
			{
				for (int k1 = 0; k1 < kernel_size_; k1++)
				{
					for (int n = 0; n < num; n++)
					{
						/*height_in equals to height_out in this case*/
						for (int h_out = 0; h_out < height_in; h_out++)
						{
							/*
							  dst dimension: (c_in * k0 * k1) * (num * h_in * w_in)
							  ori position: n, c_in, h_in + k0, k1
							  dst position: c_in, k0, k1, n, h_in, 0
							*/
							memcpy(data_matrix + ((((c_in * kernel_size_ + k0) * kernel_size_ + k1) * num + n) * height_in + h_out) * width_in,
								buffer_in + ((n * channel_in + c_in) * height_pad + h_out + k0) * width_pad + k1,
								width_in * sizeof(data_type));
						}
					}
				}
			}
		}
	}
	/*for loop method, slow*/
	else
	{
		for (int n = 0; n < num; n++)
		{
			for (int h_out = 0; h_out < height_out; h_out++)
			{
				for (int w_out = 0; w_out < width_out; w_out++)
				{
					for (int k0 = 0; k0 < kernel_size_; k0++)
					{
						for (int k1 = 0; k1 < kernel_size_; k1++)
						{
							for (int c_in = 0; c_in < channel_in; c_in++)
							{
								if (h_out == 1 && w_out == 488)
								{
									int qqq = 3;
								}
								/*
								  ori dimension: num, c_in, height_in, width_in
								  matrix dimension: (c_in * k0 * k1) * (num * height_in * width_in)
								*/
								data_matrix[((((c_in * kernel_size_ + k0) * kernel_size_ + k1) * num + n) * height_out + h_out) * width_out + w_out]
									= buffer_in[((n * channel_in + c_in) * height_pad + padding_ + h_out * stride_ + k0 - kernel_size_ / 2) * width_pad + padding_ + w_out * stride_ + k1 - kernel_size_ / 2];
							}
						}
					}
				}
			}
		}
	}
	finish = clock();
	//cout << finish - start << ":mat transfer" << endl;
	

	start = clock();
	/*(dim0 * dim1) * (dim1 * dim2)*/
	int dim0 = channel_out;
	int dim1 = kernel_size_ * kernel_size_ * channel_in;
	int dim2 = height_out * width_out * num;
	data_type *out_matrix = new data_type[dim0 * dim2];
	matrix_multiply(weight_matrix, data_matrix, out_matrix, dim0, dim1, dim2);
	/*for (int d0 = 0; d0 < dim0; d0++)
	{
		for (int d2 = 0; d2 < dim2; d2++)
		{
			data_type cur_val = 0;
			for (int d1 = 0; d1 < dim1; d1++)
			{
				cur_val += weight_matrix[d0 * dim1 + d1] * data_matrix[d1 * dim2 + d2];
			}
			out_matrix[d0 * dim2 + d2] = cur_val;
		}
	}*/
	finish = clock();
	//cout << finish - start << ": matrix_mult" << endl;

	start = clock();
	/*transfer to output tensor*/
	memcpy(top_tensor->get_data_ptr()->data_ptr(), out_matrix, dim0 * dim2 * sizeof(data_type));


	delete[] buffer_in;
	delete[] buffer_weight;
	delete[] data_matrix;
	delete[] weight_matrix;
	delete[] out_matrix;
	finish = clock();
	//cout << finish - start << ": memcpy" << endl;
	return;

	/*for loop realization*/
	start = clock();
	int iter_num = 0;
	for (int n = 0; n < num; n++)
	{
		for (int c_out = 0; c_out < channel_out; c_out++)
		{
			//int h_phase = -padding_ + kernel_size_ / 2;
			int h_phase = kernel_size_ / 2;
			for (int h = 0; h < height_out; h++)
			{
				//int w_phase = -padding_ + kernel_size_ / 2;
				int w_phase = kernel_size_ / 2;
				for (int w = 0; w < width_out; w++)
				{		
					data_type out_val = 0; 
					if (n == 0 && h == 10 && w == 10 && c_out == 0)
					{
						int qqq = 3;
					}
					for (int i = -kernel_size_ / 2; i <= kernel_size_ / 2; i++)
					{
						for (int j = -kernel_size_ / 2; j <= kernel_size_ / 2; j++)
						{
							for (int c_in = 0; c_in < channel_in; c_in++)
							{
								data_type cur_val = buffer_in[((n * channel_in + c_in) * height_pad + h_phase + i) * width_pad + w_phase + j];
								data_type cur_w = buffer_weight[(((i + kernel_size_ / 2) * kernel_size_ + j + kernel_size_ / 2) * channel_in + c_in) * channel_out + c_out];
								out_val += cur_val * cur_w;
							}
						}
					}
					/*out buffer will not speedup because iter time is equal*/
					top_tensor->set_data(n, c_out, h, w, out_val);
					iter_num++;
					w_phase += stride_;
				}
				h_phase += stride_;
			}
		}
	}
	finish = clock();
	//cout << finish - start << ": time" << endl;
	//cout << iter_num << ": iter" << endl;

	delete[] buffer_in;
	delete[] buffer_weight;
	//delete[] data_matrix;
	//delete[] weight_matrix;
	//delete[] out_matrix;
}

template <typename data_type>
void conv_layer_c<data_type>::matrix_multiply(data_type *mat_in0, data_type *mat_in1, data_type *mat_out, int dim0, int dim1, int dim2)
{
	int cut_num = 10;
	cut_num = max(1, dim2 / 1000);
	/*start column in mat_in1*/
	int col_index = 0;
	for (int cut_index = 0; cut_index < cut_num; cut_index++)
	{
		int cut_dim2 = dim2 / cut_num;
		if (cut_index == cut_num - 1)
			cut_dim2 += dim2%cut_num;
		data_type *cut_mat1 = new data_type[dim1 * cut_dim2];
		data_type *cut_mat_out = new data_type[dim0 * cut_dim2];
		/*cut mat_in1 into cut_mat1*/
		for (int d1 = 0; d1 < dim1; d1++)
		{
			memcpy(cut_mat1 + d1 * cut_dim2, mat_in1 + d1 * dim2 + col_index, cut_dim2 * sizeof(data_type));
		}
		/*multiply mat_in0 and cut_mat1*/
		for (int d0 = 0; d0 < dim0; d0++)
		{
			for (int d2 = 0; d2 < cut_dim2; d2++)
			{
				data_type cur_val = 0;
				for (int d1 = 0; d1 < dim1; d1++)
				{
					cur_val+= mat_in0[d0 * dim1 + d1] * cut_mat1[d1 * cut_dim2 + d2];
				}
				cut_mat_out[d0 * cut_dim2 + d2] = cur_val;
			}
		}
		/*put cut_mat_out into mat_out*/
		for (int d0 = 0; d0 < dim0; d0++)
		{
			memcpy(mat_out + d0 * dim2 + col_index, cut_mat_out + d0 * cut_dim2, cut_dim2 * sizeof(data_type));
		}

		col_index += cut_dim2;
		delete[] cut_mat1;
		delete[] cut_mat_out;
	}


	/*normal matrix multiply*/
	//int index0, index1;
	//for (int d0 = 0; d0 < dim0; d0++)
	//{
	//	for (int d2 = 0; d2 < dim2; d2++)
	//	{
	//		data_type cur_val = 0;
	//		index1 = 0;
	//		for (int d1 = 0; d1 < dim1; d1++)
	//		{
	//			//cur_val += mat_in0[d0 * dim1 + d1] * mat_in1[d1 * dim2 + d2];
	//			cur_val += mat_in0[d0 * dim1 + d1] * mat_in1[index1];
	//			index1 +=dim2/2.5;
	//		}
	//		mat_out[d0 * dim2 + d2] = cur_val;
	//	}
	//}
}

INSTANTIATE_CLASS(conv_layer_c);