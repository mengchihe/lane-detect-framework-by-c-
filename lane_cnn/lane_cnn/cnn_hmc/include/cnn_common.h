/*include all common operation in cnn*/
#ifndef CNN_COMMON_H
#define CNN_COMMON_H

#include "../../common.h"

#define DEBUG

//can't add brackets?????????????????????
#define INSTANTIATE_CLASS(class_name) \
	template class class_name<double>;\
	template class class_name<int>;\
	template class class_name<unsigned int>;\
	template class class_name<float>;\
	template class class_name<byte>

#endif