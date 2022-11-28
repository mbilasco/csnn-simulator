#include "process/GrayScale.h"

using namespace process;

static RegisterClassParameter<GrayScale, ProcessFactory> _register_1("GrayScale");


GrayScale::GrayScale() : UniquePassProcess(_register_1),
	_width(0), _height(0), _depth(0) {

}

Shape GrayScale::compute_shape(const Shape& shape) {
	_width = shape.dim(0);
	_height = shape.dim(1);
	_depth = 1;
	return Shape({_width, _height, _depth});
}

void GrayScale::process_train(const std::string&, Tensor<float>& sample) {
	_process(sample);
}

void GrayScale::process_test(const std::string&, Tensor<float>& sample) {
	_process(sample);
}

void GrayScale::_process(Tensor<float>& in) const {

	Tensor<float> out(Shape({_width, _height, _depth}));

	for(size_t x=0; x<_width; x++) {
		for(size_t y=0; y<_height; y++) {
			float v = 0;
            for(size_t z=0; z<in.shape().dim(2); z++) {
                v += in.at(x,y,z) * in.at(x,y,z);
			}
            out.at(x, y, 1) = sqrt(v / 3);
		}
	}

	in = out;
}
