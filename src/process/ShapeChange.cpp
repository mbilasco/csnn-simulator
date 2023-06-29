#include "process/ShapeChange.h"

using namespace process;

//
//	ShapeChange a class that only changes the expected output shape of the data. This is useful in case you want to test something like On-off filtering with the SVM.cpp class.
//
static RegisterClassParameter<ShapeChange, ProcessFactory> _register_1("ShapeChange");

ShapeChange::ShapeChange() : UniquePassProcess(_register_1), _width(0), _height(0), _depth(0), _temporal_depth(0)
{
}

ShapeChange::ShapeChange(size_t width, size_t height, size_t depth, size_t temporal_depth) : ShapeChange()
{
	_height = height;
	_width =  width;
	_depth = depth;
	_temporal_depth = temporal_depth;
}

void ShapeChange::process_train(const std::string &label, Tensor<float> &sample)
{
	_process(label, sample);
}

void ShapeChange::process_test(const std::string &label, Tensor<float> &sample)
{
	_process(label, sample);
}

Shape ShapeChange::compute_shape(const Shape &shape)
{
	return Shape({_height, _width, _depth, _temporal_depth});
}

void ShapeChange::_process(const std::string &label, Tensor<InputType> &in) const
{
	
}