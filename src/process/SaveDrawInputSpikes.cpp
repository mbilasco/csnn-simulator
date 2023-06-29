#include "process/SaveDrawInputSpikes.h"

using namespace process;

//
//	SaveDrawInputSpikes
//
static RegisterClassParameter<SaveDrawInputSpikes, ProcessFactory> _register_1("SaveDrawInputSpikes");

SaveDrawInputSpikes::SaveDrawInputSpikes() : UniquePassProcess(_register_1), _expName(""), _width(0), _height(0), _depth(0), _conv_depth(0), _save(0), _draw(0)
{
}

SaveDrawInputSpikes::SaveDrawInputSpikes(std::string expName, size_t save, size_t draw) : SaveDrawInputSpikes()
{
	_save = save;
	_draw = draw;
	_expName = expName;
}

void SaveDrawInputSpikes::process_train(const std::string &label, Tensor<float> &sample)
{
	_process(label, sample);
}

void SaveDrawInputSpikes::process_test(const std::string &label, Tensor<float> &sample)
{
	_process(label, sample);
}

Shape SaveDrawInputSpikes::compute_shape(const Shape &shape)
{
	_height = shape.dim(0);
	_width = shape.dim(1);
	_depth = shape.dim(2);
	_conv_depth = shape.dim(3);; 
	return Shape({_height, _width, _depth, _conv_depth});
}

void SaveDrawInputSpikes::_process(const std::string &label, Tensor<InputType> &in) const
{
}