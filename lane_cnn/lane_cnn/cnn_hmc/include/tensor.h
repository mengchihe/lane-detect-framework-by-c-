#ifndef TENSOR_H
#define TENSOR_H

#include "cnn_common.h"
#include "memory.h"

template <typename data_type>
class tensor_c
{
public:
	tensor_c():data_(), shape_() {};
	explicit tensor_c(int num, int channel, int height, int width);
	explicit tensor_c(int num, int channel, int height, int width, string name);
	explicit tensor_c(vector<int> shape);
	explicit tensor_c(vector<int> shape, string name);
	/*initial input tensor from a memory buffer*/
	explicit tensor_c(byte * buffer_in, int num, int channel, int height, int width, string name);
	~tensor_c();

	shared_ptr<memory_c<data_type>> get_data_ptr();
	inline data_type get_data(int n, int c, int h, int w);
	inline data_type get_data(int pos);
	inline void set_data(int n, int c, int h, int w, data_type val);
	inline void set_data(int pos, data_type val);
	void reshape(int num, int channel, int height, int width);
	inline vector<int> get_shape();
	string get_name();
	void copy_from(tensor_c<data_type> tensor);
	/*copy tensor's data of one num into current tensor's offset position*/
	void copy_data(tensor_c<data_type> tensor, int offset, int num);
	/*copy fetch tensor into output buffer*/
	void to_mem_buff(byte *buffer_out);
	/*pre processing for input data*/
	void data_pre_process();

	/*get the offset to start pos in memory_c when set/get data*/
	inline int offset(int pos);
	inline int offset(int n, int c, int h, int w);

protected:
	shared_ptr<memory_c<data_type>> data_;
	vector<int> shape_;
	string name_;
};

#endif