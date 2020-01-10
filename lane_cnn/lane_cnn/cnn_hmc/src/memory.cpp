#include "../include/memory.h"

template <typename data_type>
memory_c<data_type>::memory_c(int length)
{
	mem_data_ = new data_type[length];
}

template <typename data_type>
memory_c<data_type>::~memory_c()
{
	delete[] mem_data_;
}

//template <typename data_type>
//data_type *memory_c<data_type>::data_ptr()
//{
//	return mem_data_;
//}

INSTANTIATE_CLASS(memory_c);