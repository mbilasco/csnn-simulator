#ifndef _PROCESS_TEMPORAL_GAUSSIAN_FILTER_H
#define _PROCESS_TEMPORAL_GAUSSIAN_FILTER_H

#include <iostream>
#include "Process.h"
#include "NumpyReader.h"

namespace process {

	namespace _priv {
		class TemporalGaussianFilterHelper {

		public:
			TemporalGaussianFilterHelper() = delete;

			static Tensor<float> generate_filter(size_t filter_size, float center_dev);

		};
	}

	/**
	 * @brief On-center/off-center filtering in a pre-processing technique that is similar to DoG filtering to detect edges.
	 * In fact, it is to extract the difference in intensity which helps the SNN encode the image as spikes.
	 * This filter puts the on and off cells in 2 different channels.
	 * 
	 * @param expName std::string - The name of the experiment, in order to choose the folder name.
	 * @param draw size_t - If set to 1, the input frames are drawn in the ReDrawInput folder in the build file. Otherwise, nothing is drawn.
	 * @param filter_size size_t - The size of the on-center/off-center filter.
	 * @param center_dev float - The variance of the Gaussian kernels DoG center.
	 */
	class TemporalGaussianFilter : public UniquePassProcess {

	public:
		TemporalGaussianFilter();
		TemporalGaussianFilter(std::string _expName, size_t _draw = 0, size_t filter_size = 1, float center_dev = 1);

		virtual Shape compute_shape(const Shape& shape);
		virtual void process_train(const std::string& label, Tensor<float>& sample);
		virtual void process_test(const std::string& label, Tensor<float>& sample);

	private:
		void _process(const std::string& label, Tensor<float>& in) const;

		size_t _draw;
		std::string _expName;
		size_t _filter_size;
		float _center_dev;
		// To save the drawings
		std::string _file_path;

		size_t _width;
		size_t _height;
		size_t _depth;
		size_t _conv_depth;
		Tensor<float> _filter;
	};	
}
#endif
