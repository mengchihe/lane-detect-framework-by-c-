#include "../include/tensor.h"


template <typename data_type>
tensor_c<data_type>::tensor_c(int num, int channel, int height, int width)
{
	shape_.push_back(num);
	shape_.push_back(channel);
	shape_.push_back(height);
	shape_.push_back(width);
	data_.reset(new memory_c<data_type>(num*channel*height*width));
}

template <typename data_type>
tensor_c<data_type>::tensor_c(int num, int channel, int height, int width, string name)
{
	shape_.push_back(num);
	shape_.push_back(channel);
	shape_.push_back(height);
	shape_.push_back(width);
	name_ = name;
	data_.reset(new memory_c<data_type>(num*height*width*channel));
}

template <typename data_type>
tensor_c<data_type>::tensor_c(vector<int> shape)
{
#ifdef DEBUG
	if (shape.size() == 0)
	{
		cout << "tensor.cpp: tensor initial dimension is zero!!!!!!!!!!!\n" << endl;
		system("pause");
	}
#endif
	shape_.assign(shape.begin(), shape.end());
	int data_size = shape[0];
	for (int i = 1; i < shape.size(); i++)
	{
		data_size *= shape[i];
	}
	data_.reset(new memory_c<data_type>(data_size));
}

template <typename data_type>
tensor_c<data_type>::tensor_c(vector<int> shape, string name)
{
	name_ = name;
#ifdef DEBUG
	if (shape.size() == 0)
	{
		cout << "tensor.cpp: tensor initial dimension is zero!!!!!!!!!!!\n" << endl;
		system("pause");
	}
#endif
	shape_.assign(shape.begin(), shape.end());
	int data_size = shape[0];
	for (int i = 1; i < shape.size(); i++)
	{
		data_size *= shape[i];
	}
	data_.reset(new memory_c<data_type>(data_size));
}

template <typename data_type>
tensor_c<data_type>::tensor_c(byte *buffer_in, int num, int channel, int height, int width, string name)
{
	name_ = name;
	shape_.push_back(num);
	shape_.push_back(channel);
	shape_.push_back(height);
	shape_.push_back(width);
	data_.reset(new memory_c<data_type>(num * height * width * channel));
	//int data_length = num * channel * height * width;
	//memcpy(data_->data_ptr(), buffer_in, data_length * sizeof(data_type));
	for (int n = 0; n < num; n++)
	{
		for (int c = 0; c < channel; c++)
		{
			for (int h = 0; h < height; h++)
			{
				for (int w = 0; w < width; w++)
				{
					if (h == 200 && w == 10)
					{
						int qqq = 3;
					}
					/*use offset func here will affect the speed fiercely*/
					data_->data_ptr()[offset(n, c, h, w)]
					//data_->data_ptr()[n*height * width * channel + h*width * channel + w * channel + c]
						= buffer_in[((n * channel + c) * height + h) * width + w];
				}
			}
		}
	}
}

template <typename data_type>
tensor_c<data_type>::~tensor_c()
{
	/*mem data will releas in memory_c*/
}

template <typename data_type>
shared_ptr<memory_c<data_type>> tensor_c<data_type>::get_data_ptr()
{
	return data_;
}

template <typename data_type>
inline data_type tensor_c<data_type>::get_data(int n, int c, int h, int w)
{
#ifdef DEBUG
	if (n < 0 || n >= shape_[0] || c < 0 || c >= shape_[1] || h < 0 || h >= shape_[2] || w < 0 || w >= shape_[3])
	{
		cout << "tensor.cpp: get data index exceed tensor's dimension!!!!!!!!!!\n" << endl;
		system("pause");
	}
#endif
	return data_->data_ptr()[offset(n, c, h, w)];
}

template <typename data_type>
inline data_type tensor_c<data_type>::get_data(int pos)
{
	return data_->data_ptr()[offset(pos)];
}

template <typename data_type>
inline void tensor_c<data_type>::set_data(int n, int c, int h, int w, data_type val)
{
#ifdef DEBUG
	if (n < 0 || n >= shape_[0] || c < 0 || c >= shape_[1] || h < 0 || h >= shape_[2] || w < 0 || w >= shape_[3])
	{
		cout << "tensor.cpp: set data index exceed tensor's dimension!!!!!!!!!!\n" << endl;
		system("pause");
	}
#endif
	data_->data_ptr()[offset(n, c, h, w)] = val;
}

template <typename data_type>
inline void tensor_c<data_type>::set_data(int pos, data_type val)
{
	data_->data_ptr()[offset(pos)] = val;
}

template <typename data_type>
void tensor_c<data_type>::reshape(int num, int channel, int height, int width)
{
	shape_.clear();
	//delete[] data_;
	shape_.push_back(num);
	shape_.push_back(channel);
	shape_.push_back(height);
	shape_.push_back(width);
	//data_.reset(new data_type[num*height*width*channel]);
	data_.reset(new memory_c<data_type>(num*height*width*channel));
}

template <typename data_type>
inline vector<int> tensor_c<data_type>::get_shape()
{
	return shape_;
}

template <typename data_type>
string tensor_c<data_type>::get_name()
{
	return name_;
}

template <typename data_type>
void tensor_c<data_type>::copy_from(tensor_c<data_type> tensor)
{
	/*
	  should not use memcpy to class, may be just copy the pointer of the vector
	  and the vector in dst class will be valid when src class is released
	*/
	//memcpy(this, &tensor, sizeof(tensor_c<data_type>));
	shape_.clear();
	shape_.push_back(tensor.shape_[0]);
	shape_.push_back(tensor.shape_[1]);
	shape_.push_back(tensor.shape_[2]);
	shape_.push_back(tensor.shape_[3]);
	name_ = tensor.name_;
	int data_len = shape_[0] * shape_[1] * shape_[2] * shape_[3];
	data_.reset(new memory_c<data_type>(data_len));
	memcpy(data_->data_ptr(), tensor.data_->data_ptr(), data_len * sizeof(data_type));
}

template <typename data_type>
void tensor_c<data_type>::copy_data(tensor_c<data_type> tensor, int offset, int num)
{
	int data_len = 1 * tensor.get_shape()[1] * shape_[2] * shape_[3];
	memcpy(data_->data_ptr() + offset, tensor.data_->data_ptr() + num * data_len, data_len * sizeof(data_type));
}

template <typename data_type>
void tensor_c<data_type>::to_mem_buff(byte *buffer_out)
{
	int num = shape_[0];
	int channel = shape_[1];
	int height = shape_[2];
	int width = shape_[3];
	memcpy(buffer_out, data_->data_ptr(), num * channel * height * width * sizeof(data_type));
	//for (int n = 0; n < num; n++)
	//{
	//	for (int h = 0; h < height; h++)
	//	{
	//		for (int w = 0; w < width; w++)
	//		{
	//			for (int c = 0; c < channel; c++)
	//			{
	//				/*use offset func here will affect the speed fiercely*/
	//				buffer_out[n * height * width * channel + c * height * width + h * width + w]
	//					//= data_->data_ptr()[offset(n, h, w, c)];
	//					= data_->data_ptr()[n*height * width * channel + h*width * channel + w * channel + c];
	//			}
	//		}
	//	}
	//}
}

template <typename data_type>
void tensor_c<data_type>::data_pre_process()
{
	data_type vgg_mean[3] = { 103.939, 116.779, 123.68 };
	int num = shape_[0];
	int channel = shape_[1];
	int height = shape_[2];
	int width = shape_[3];
	for (int n = 0; n < num; n++)
	{
		for (int c = 0; c < channel; c++)
		{
			for (int h = 0; h < height; h++)
			{
				for (int w = 0; w < width; w++)
				{
					if (n == 0 && h == 200 && w == 100 && c == 0)
					{
						int qqq = 3;
					}
					data_type cur_val = this->get_data(n, c, h, w);
					cur_val -= vgg_mean[c];
					this->set_data(n, c, h, w, cur_val);
				}
			}
		}
	}
}

template <typename data_type>
inline int tensor_c<data_type>::offset(int pos)
{
#ifdef DEBUG
	if (pos >= shape_[0] * shape_[1] * shape_[2] * shape_[3])
	{
		cout << "tensor.cpp: offset exceed data length!!!!!!!!!!\n" << endl;
		system("pause");
	}
#endif
	return pos;
}

template <typename data_type>
inline int tensor_c<data_type>::offset(int n, int c, int h, int w)
{
#ifdef DEBUG
	if (n >= shape_[0] || c >= shape_[1] || h >= shape_[2] || w >= shape_[3])
	{
		cout << "tensor.cpp: offset exceed dimension!!!!!!!!!!\n" << endl;
		system("pause");
	}
#endif
	return ((n * shape_[1] + c) * shape_[2] + h) * shape_[3] + w;
}


INSTANTIATE_CLASS(tensor_c);