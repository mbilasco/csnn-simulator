#include "process/TemporalGaussianFilter.h"

using namespace process;

#include "Experiment.h"
#include "Math.h"

//
//	TemporalGaussianFilter
//

static RegisterClassParameter<TemporalGaussianFilter, ProcessFactory> _register_1("TemporalGaussianFilter");

TemporalGaussianFilter::TemporalGaussianFilter() : UniquePassProcess(_register_1),
												   _expName(""), _draw(0), _filter_size(0), _center_dev(0), _height(0), _width(0), _depth(0), _conv_depth(0), _filter()
{
	add_parameter("draw", _draw);
	add_parameter("_filter_size", _filter_size);
	add_parameter("center_dev", _center_dev);
}

TemporalGaussianFilter::TemporalGaussianFilter(std::string expName, size_t draw, size_t _filter_size, float center_dev) : TemporalGaussianFilter()
{
	parameter<size_t>("draw").set(draw);
	_expName = expName;
	parameter<size_t>("_filter_size").set(_filter_size);
	parameter<float>("center_dev").set(center_dev);
	if (draw == 1)
		std::filesystem::create_directories("Input_frames/" + _expName + "/TGF/");

	_file_path = std::filesystem::current_path();
}

Shape TemporalGaussianFilter::compute_shape(const Shape &shape)
{
	parameter<size_t>("_filter_size").ensure_initialized(experiment()->random_generator());
	parameter<float>("center_dev").ensure_initialized(experiment()->random_generator());

	_height = shape.dim(0);
	_width = shape.dim(1);
	_depth = shape.dim(2);
	_conv_depth = shape.dim(3) > 3 ? shape.dim(3) : 1;
	return Shape({_height, _width, _depth, _conv_depth});
}

void TemporalGaussianFilter::process_train(const std::string &label, Tensor<float> &sample)
{
	_process(label, sample);
}

void TemporalGaussianFilter::process_test(const std::string &label, Tensor<float> &sample)
{
	_process(label, sample);
}

void TemporalGaussianFilter::_process(const std::string &label, Tensor<InputType> &in) const
{

	Tensor<float> t2(Shape({_filter_size}));
	/**
	 * @brief t2 is the value of (k^2) in the case of 1D temporal filtering.
	 */
	for (size_t k = 0; k < _filter_size; k++)
		t2.at(k) = std::pow((k + 1) - static_cast<float>(_filter_size) / 2.0f - 0.5f, 2.0f);

	Tensor<float> filter(Shape({_filter_size}));
	float filter_sum = 0;

	for (size_t k = 0; k < _filter_size; k++)
	{
		filter.at(k) = 1.0f / std::sqrt(2.0f * static_cast<float>(M_PI)) * (1.0f / _center_dev * std::exp(-t2.at(k) / 2.0f / (_center_dev * _center_dev)));
		filter_sum += filter.at(k);
	}

	for (size_t k = 0; k < _filter_size; k++)
		filter.at(k) /= filter_sum;

	if (in.shape().number() <= 3)
		throw std::runtime_error("A temporal filter can only be applied when the data has a temporal depth > 1");

	else
	{
		Tensor<InputType> out(Shape({_height, _width, _depth, _conv_depth}));
		for (size_t x = 0; x < _height; x++)
			for (size_t y = 0; y < _width; y++)
				for (size_t z = 0; z < _depth; z++)
					for (size_t k = 0; k < _conv_depth; k++)
					{
						float v = 0;
						for (size_t fk = 0; fk < _filter_size; fk++)
						{
							size_t k_in = k + fk > _filter_size / 2 ? std::min(k + fk - _filter_size / 2, _conv_depth - 1) : 0;
							v += in.at(x, y, z, k_in) * filter.at(fk);
						}

						out.at(x, y, z, k) = v;
					}

		if (_draw == 1)
			Tensor<float>::draw_nonscaled_tensor(_file_path + "/Input_frames/" + _expName + "/TGF/TGF_" + label + "_", out);
		in = out;
	}
}