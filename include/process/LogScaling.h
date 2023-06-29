#ifndef _PROCESS_Log_SCALING_H
#define _PROCESS_Log_SCALING_H

#include <filesystem>
#include <iostream>
#include "Process.h"
#include "NumpyReader.h"

namespace process
{

	namespace _priv
	{
		class LogScaling
		{

		public:
			LogScaling() = delete;
		};
	}
	/**
	 * @brief This method divides the value of each pixel by the Logimum pixel value. This is done for scaling.
	 *
	 */
	class LogScaling : public UniquePassProcess
	{

	public:
		LogScaling();

		virtual Shape compute_shape(const Shape &shape);
		virtual void process_train(const std::string &label, Tensor<float> &sample);
		virtual void process_test(const std::string &label, Tensor<float> &sample);

	private:
		void _process(const std::string &label, Tensor<float> &in) const;

		size_t _width;
		size_t _height;
		size_t _depth;
		size_t _conv_depth;
	};

}
#endif
