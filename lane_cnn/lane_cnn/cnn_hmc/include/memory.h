#ifndef MEMORY_H
#define MEMORY_H

#include "cnn_common.h"

template <typename data_type>
class memory_c
{
public:
	memory_c() : mem_data_(), length_() {};
	explicit memory_c(int length);
	~memory_c();

	inline data_type* data_ptr() const
	{
		return mem_data_;
	}
protected:
	data_type* mem_data_;
	int length_;
};

#endif