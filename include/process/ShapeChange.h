#ifndef _PROCESS_SHAPE_CHANGE_H
#define _PROCESS_SHAPE_CHANGE_H

#include <filesystem>
#include <iostream>
#include "Process.h"
#include "NumpyReader.h"

namespace process
{

	namespace _priv
	{
		class ShapeChange
		{

		public:
			ShapeChange() = delete;
		};
	}
	/**
	 * @brief This method divides each frame pixel by the maximum pixel value. This is done for scaling.
	 * 
	 * @param expName The name of the expirement, in order to choose the folder name.
	 * @param draw If set to 1, the input frames are drawn in the ReDrawInput folder in the build file. Otherwise, nothing is drawn.
	 * @param scalar A constant to scale the input.
	 */
	class ShapeChange : public UniquePassProcess
	{

	public:
		ShapeChange();
		ShapeChange(size_t width, size_t height, size_t depth, size_t temporal_depth);

		virtual Shape compute_shape(const Shape &shape);
		virtual void process_train(const std::string &label, Tensor<float> &sample);
		virtual void process_test(const std::string &label, Tensor<float> &sample);

	private:
		void _process(const std::string &label, Tensor<float> &in) const;

		size_t _width;
		size_t _height;
		size_t _depth;
		size_t _temporal_depth;
	};

}
#endif
