#include "process/GaussianFilter.h"

using namespace process;

#include "Experiment.h"
#include "Math.h"

//
//	GaussianFilter
//

static RegisterClassParameter<GaussianFilter, ProcessFactory> _register_1("GaussianFilter");

GaussianFilter::GaussianFilter() : UniquePassProcess(_register_1),
								   _expName(""), _draw(0), _filter_size(0), _center_dev(0), _height(0), _width(0), _depth(0), _conv_depth(0), _filter()
{
	add_parameter("draw", _draw);
	add_parameter("_filter_size", _filter_size);
	add_parameter("center_dev", _center_dev);
}

GaussianFilter::GaussianFilter(std::string expName, size_t draw, size_t _filter_size, float center_dev) : GaussianFilter()
{
	parameter<size_t>("draw").set(draw);
	_expName = expName;
	parameter<size_t>("_filter_size").set(_filter_size);
	parameter<float>("center_dev").set(center_dev);
	if (draw == 1)
		std::filesystem::create_directories("Input_frames/" + _expName + "/GF/");

	_file_path = std::filesystem::current_path();
}

Shape GaussianFilter::compute_shape(const Shape &shape)
{
	parameter<size_t>("_filter_size").ensure_initialized(experiment()->random_generator());
	parameter<float>("center_dev").ensure_initialized(experiment()->random_generator());

	_height = shape.dim(0);
	_width = shape.dim(1);
	_depth = shape.dim(2);
	_conv_depth = shape.dim(3) > 3 ? shape.dim(3) : 1;
	return Shape({_height, _width, _depth, _conv_depth});
}

void GaussianFilter::process_train(const std::string &label, Tensor<float> &sample)
{
	_process(label, sample);
}

void GaussianFilter::process_test(const std::string &label, Tensor<float> &sample)
{
	_process(label, sample);
}

void GaussianFilter::_process(const std::string &label, Tensor<InputType> &in) const
{
	Tensor<float> d2(Shape({_filter_size, _filter_size}));

	for (size_t i = 0; i < _filter_size; i++)
		for (size_t j = 0; j < _filter_size; j++)
		{
			d2.at(i, j) = std::pow((i + 1) - static_cast<float>(_filter_size) / 2.0f - 0.5f, 2.0f) + std::pow((j + 1) - static_cast<float>(_filter_size) / 2.0f - 0.5f, 2.0f);
		}
		
	Tensor<float> filter(Shape({_filter_size, _filter_size}));
	float filter_sum = 0;

	for (size_t x = 0; x < _filter_size; x++)
		for (size_t y = 0; y < _filter_size; y++)
		{
			filter.at(x, y) = (1.0f / (2.0f * static_cast<float>(M_PI))) * (1.0f / (_center_dev * _center_dev)) * (std::exp(-d2.at(x, y) / 2.0f / (_center_dev * _center_dev)));
			filter_sum += filter.at(x, y);
		}

	for (size_t x = 0; x < _filter_size; x++)
	 	for (size_t y = 0; y < _filter_size; y++)
	 		filter.at(x, y) /= filter_sum;

	if (in.shape().number() <= 3)
	{
		Tensor<InputType> out(Shape({_height, _width, _depth}));
		for (size_t x = 0; x < _height; x++)
			for (size_t y = 0; y < _width; y++)
				for (size_t z = 0; z < _depth; z++)
				{
					float v = 0;
					for (size_t fx = 0; fx < _filter_size; fx++)
						for (size_t fy = 0; fy < _filter_size; fy++)
						{
							size_t x_in = x + fx > _filter_size / 2 ? std::min(x + fx - _filter_size / 2, _height - 1) : 0;
							size_t y_in = y + fy > _filter_size / 2 ? std::min(y + fy - _filter_size / 2, _width - 1) : 0;
							v += in.at(x_in, y_in, z) * filter.at(fx, fy);
						}
					out.at(x, y, z) = v;
				}

		if (_draw == 1)
			Tensor<float>::draw_nonscaled_tensor(_file_path + "/Input_frames/" + _expName + "/GF/GF_" + label + "_", out);
		in = out;
	}
	else
	{
		Tensor<InputType> out(Shape({_height, _width, _depth, _conv_depth}));
		for (size_t x = 0; x < _height; x++)
			for (size_t y = 0; y < _width; y++)
				for (size_t z = 0; z < _depth; z++)
					for (size_t k = 0; k < _conv_depth; k++)
					{
						float v = 0;
						for (size_t fx = 0; fx < _filter_size; fx++)
							for (size_t fy = 0; fy < _filter_size; fy++)
							{
								size_t x_in = x + fx > _filter_size / 2 ? std::min(x + fx - _filter_size / 2, _height - 1) : 0;
								size_t y_in = y + fy > _filter_size / 2 ? std::min(y + fy - _filter_size / 2, _width - 1) : 0;

								v += in.at(x_in, y_in, z, k) * filter.at(fx, fy);
							}
						out.at(x, y, z, k) = v;
					}

		if (_draw == 1)
			Tensor<float>::draw_nonscaled_tensor(_file_path + "/Input_frames/" + _expName + "/GF/GF_" + label + "_", out);
		in = out;
	}
}