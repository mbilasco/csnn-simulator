#include "dataset/ImageBin.h"

using namespace dataset;


ImageBin::ImageBin(const std::string& image_filename, const std::string& label_filename, int width, int height, int depth, const std::string& dataset_name) :
	_image_filename(image_filename), _label_filename(label_filename),
	_image_file(image_filename, std::ios::in | std::ios::binary), _label_file(label_filename, std::ios::in | std::ios::binary),
	_shape({width, height, depth}), _next_label(0), _name(dataset_name) {

	if(!_image_file.is_open()) {
		throw std::runtime_error("Can't open "+image_filename);
	}
	if(!_label_file.is_open()) {
		throw std::runtime_error("Can't open "+label_filename);
	}

	_prepare_next();
}

bool ImageBin::has_next() const {
	return !_label_file.eof();
}


std::pair<std::string, Tensor<InputType>> ImageBin::next() {
	std::pair<std::string, Tensor<InputType>> out(std::to_string(static_cast<size_t>(_next_label)), _shape);

	for(size_t x=0; x<_shape.dim(0); x++) {
		for(size_t y=0; y<_shape.dim(1); y++) {
			for(size_t z=0; z<_shape.dim(2); z++) {
				uint8_t pixel;
				_image_file.read((char*)&pixel, sizeof(uint8_t));

				out.second.at(x, y, z) = static_cast<InputType>(pixel)/static_cast<InputType>(std::numeric_limits<uint8_t>::max());
			}
		}
	}

	_prepare_next();

	return out;
}

void ImageBin::reset() {
	_label_file.seekg(0, std::ios::beg);
	_image_file.seekg(0, std::ios::beg);
	_prepare_next();
}


void ImageBin::close() {
	_label_file.close();
	_image_file.close();
}

size_t ImageBin::size() const {
	return 0;
}

std::string ImageBin::to_string() const {
	return _name+"("+_image_filename+", "+_label_filename+")";
}

const Shape& ImageBin::shape() const {
	return _shape;
}

void ImageBin::_prepare_next() {
	_label_file.read((char*)&_next_label, sizeof(uint8_t));
}
