#ifndef _MULTIPLESTREAM_H
#define _MULTIPLESTREAM_H

#include <filesystem>
#include <cassert>
#include <limits>
#include <tuple>

#include "Input.h"
#include "tool/Operations.h"

namespace dataset
{

	/**
	 * @brief This class monitors introducing the dataset into the program, 
	 * It is responsible for counting the number of samples. This is the first step, even before transforming the data into spikes.
	 * 
	 * @param folder_path the path to the saved featuremaps
	 * @param method the feature map fusion methods 1- Concatination 2- Averaging
	 * @param draw a flag to draw the fused features
	 */
	class MultipleStream : public Input
	{

	public:
		MultipleStream(const std::string &folder_path, const size_t &method = 1, const size_t &draw = 0, size_t max_read = std::numeric_limits<size_t>::max());

		virtual bool has_next() const;
		virtual std::pair<std::string, Tensor<InputType>> next();
		virtual void reset();
		virtual void close();

		size_t size() const;
		virtual std::string to_string() const;

		virtual const Shape &shape() const;

	private:
		uint32_t swap(uint32_t v);

		std::string _folder_path;
		uint32_t _method;
		uint32_t _draw;

		std::vector<std::pair<std::string, Tensor<float>>> _features_fused;
		
		std::vector<std::string> _data_list;

		uint32_t _size;
		uint32_t _cursor;

		Shape _shape;

		uint32_t _max_read;
	};

}

#endif
