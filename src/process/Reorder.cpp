#include "process/Reorder.h"

using namespace process;

static RegisterClassParameter<UniformReorder, ProcessFactory> _registerUReorder("UniformReorder");

UniformReorder::UniformReorder() : UniquePassProcess(_registerUReorder),_min_time(0.1f),_max_time(0.9f)
{
	add_parameter("min_time", _min_time);
	add_parameter("max_time", _max_time);
}

UniformReorder::UniformReorder(float min_time, float max_time) : UniformReorder()
{
	parameter<float>("min_time").set(min_time);
	parameter<float>("max_time").set(max_time);
}

Shape UniformReorder::compute_shape(const Shape &shape)
{
	return Shape({shape.dim(0),shape.dim(1),shape.dim(2),shape.number() > 3 ? shape.dim(3) : 1});
}

void UniformReorder::process_train(const std::string &, Tensor<float> &sample)
{
	_process(sample);
}

void UniformReorder::process_test(const std::string &, Tensor<float> &sample)
{
	_process(sample);
}

void UniformReorder::_process(Tensor<float> &in) const
{

	float min_time=1;
	float max_time=0;

	for (size_t x = 0; x < in.shape().dim(0); x++)
	{
		for (size_t y = 0; y < in.shape().dim(1); y++)
		{
			for (size_t z = 0; z < in.shape().dim(2); z++)
				for (size_t k = 0; k < in.shape().dim(3); k++)
				{
					float v = in.at(x,y,z,k);
					if (v<min_time) min_time=v;
					if (v>max_time) max_time=v;

				}
		}
	}

	for (size_t x = 0; x < in.shape().dim(0); x++)
	{
		for (size_t y = 0; y < in.shape().dim(1); y++)
		{
			for (size_t z = 0; z < in.shape().dim(2); z++)
				for (size_t k = 0; k < in.shape().dim(3); k++)
				{
					float v = in.at(x,y,z,k);
					in.at(x, y, z, k) = this->_min_time + (v-min_time)/(max_time-min_time) * (this->_max_time - this->_min_time);
				}
		}
	}
}
