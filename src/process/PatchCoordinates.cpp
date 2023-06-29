#include "process/PatchCoordinates.h"

using namespace process;

//
//	PatchCoordinates
//
static RegisterClassParameter<PatchCoordinates, ProcessFactory> _register_1("PatchCoordinates");

PatchCoordinates::PatchCoordinates() : UniquePassProcess(_register_1), _width(0), _height(0), _depth(0), _conv_depth(0)
{
}

void PatchCoordinates::process_train(const std::string &label, Tensor<float> &sample)
{
	_process(label, sample);
}

void PatchCoordinates::process_test(const std::string &label, Tensor<float> &sample)
{
	// _process(label, sample);
}

Shape PatchCoordinates::compute_shape(const Shape &shape)
{
	_height = shape.dim(0);
	_width = shape.dim(1);
	_depth = shape.dim(2);
	_conv_depth = shape.dim(3);

	return Shape({_height, _width, _depth, _conv_depth});
}

void PatchCoordinates::_process(const std::string &label, Tensor<InputType> &in) const
{
	for (size_t k = 0; k < _conv_depth; k++)
		for (size_t z = 0; z < _depth; z++)
			for (size_t y = 0; y < _height; y++)
				for (size_t x = 0; x < _width; x++)
				{
					if (in.at(y, x, z, k) != INFINITE_TIME)
						set_spike_coordinates(std::tuple<size_t, size_t, size_t, size_t>{y, x, 0, k});
				}
}