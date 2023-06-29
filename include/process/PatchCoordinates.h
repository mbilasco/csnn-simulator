#ifndef _PROCESS_PATCH_COORDINATES_H
#define _PROCESS_PATCH_COORDINATES_H

#include <filesystem>
#include <iostream>
#include "Process.h"
#include "NumpyReader.h"
#include "tool/Operations.h"

namespace process
{
	namespace _priv
	{
		class PatchCoordinates
		{

		public:
			PatchCoordinates() = delete;
		};
	}
	/**
	 * @brief This method is used to collect the spike coordinates to provide some kind of attention to the chosen patches during training.
	 *
	 */
	class PatchCoordinates : public UniquePassProcess
	{

	public:
		PatchCoordinates();

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
