#ifndef _PROCESS_REORDER_H
#define _PROCESS_REORDER_H

#include "Process.h"
#include "tool/Operations.h"

namespace process
{
	static int _train_sample_count = 0;
	static int _test_sample_count = 0;

	/**
	 * @brief A type of pooling that reduces the size of the input sample by averaging the values of the each set of pixels in the pooling filter. In both the spatial and temporal dimensions.
	 * @param target_width The desired output width after pooling
	 * @param target_height The desired output height after pooling
	 * @param target_conv_depth The desired output depth after pooling
	 */
	class UniformReorder : public UniquePassProcess
	{

	public:
		UniformReorderReorder();
		UniformReorderReorder(float min_time, float max_time);

		virtual Shape compute_shape(const Shape &shape);
		virtual void process_train(const std::string &label, Tensor<float> &sample);
		virtual void process_test(const std::string &label, Tensor<float> &sample);

	private:
		void _process(Tensor<float> &sample) const;

		size_t _min_time;
		size_t _max_time;

	};

}
