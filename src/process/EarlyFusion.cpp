#include "process/EarlyFusion.h"

using namespace process;

//
//	EarlyFusion
//
static RegisterClassParameter<EarlyFusion, ProcessFactory> _register_1("EarlyFusion");

EarlyFusion::EarlyFusion() : UniquePassProcess(_register_1),
							 _expName(""), _draw(0), _fused_frames_number(0), _method(0), _width(0), _height(0), _depth(0), _conv_depth(0)
{
	add_parameter("draw", _draw);
	add_parameter("fused_frames_number", _fused_frames_number);
	add_parameter("method", _method);
}

EarlyFusion::EarlyFusion(std::string expName, size_t draw, size_t fused_frames_number, size_t method) : EarlyFusion()
{
	parameter<size_t>("draw").set(draw);
	parameter<size_t>("method").set(method);
	_expName = expName;
	_method = method;
	parameter<size_t>("fused_frames_number").set(fused_frames_number);
	_fused_frames_number = fused_frames_number;
	if (draw == 1)
	{
		std::filesystem::create_directories("Input_frames/" + _expName + "/EF/");
		_file_path = std::filesystem::current_path();
	}
}

void EarlyFusion::process_train(const std::string &label, Tensor<float> &sample)
{
	_process(label, sample);
}

void EarlyFusion::process_test(const std::string &label, Tensor<float> &sample)
{
	_process(label, sample);
}

// The -1 is because this process looses one frame by getting the differences of 2 frames.
Shape EarlyFusion::compute_shape(const Shape &shape)
{
	_height = shape.dim(0);
	_width = shape.dim(1);
	_depth = shape.dim(2);
	_conv_depth = shape.dim(3);

	if (_fused_frames_number == 0)
		_fused_frames_number = _conv_depth;

	if (_conv_depth < 2)
	{
		throw std::runtime_error("Set the _fused_frames_number variable to a number >= 2 in your early fusion function");
	}

	if (_method == 0)
		return Shape({_height * _fused_frames_number, _width, _depth, 1});
	else
		return Shape({_height, _width * _fused_frames_number, _depth, 1});
}


void EarlyFusion::_process(const std::string &label, Tensor<InputType> &in) const
{

	// The frames to fuse.
	std::vector<cv::Mat> _frames;
	// This function returns the sequence of frames as Mats.
	if (in.shape().dim(2) >= 3)
		Tensor<float>::tensor_to_colored_matrices(_frames, in);
	else
		Tensor<float>::tensor_to_matrices(_frames, in);

	if (in.shape().dim(2) >= 3)
	{
		if (_method == 0)
		{ // The output fused tensor
			Tensor<InputType> out(Shape({_height * _fused_frames_number, _width, _depth, _conv_depth / _fused_frames_number}));
			cv::Mat totalframe(_height * _fused_frames_number, _width, CV_32FC3);
			for (int _i = 0; _i < _fused_frames_number; _i++)
			{
				cv::Mat _frame = _frames[_i];
				for (int r = 0; r < _height; r++) // with each loop take a new line
					_frame.row(r).copyTo(totalframe.row(r * _fused_frames_number + _i));
			}
			Tensor<float>::matrix_to_colored_f_tensor(totalframe, out);
			totalframe = cv::Mat::zeros(_height * _fused_frames_number, _width, _depth);
			if (_draw == 1)
				Tensor<float>::draw_colored_tensor(_file_path + "/Input_frames/" + _expName + "/EF/EF_" + label + "_", out);

			in = out;
		}
		else if (_method == 1)
		{ // The output fused tensor
			Tensor<InputType> out(Shape({_height, _width * _fused_frames_number, _depth, _conv_depth / _fused_frames_number}));

			cv::Mat totalframe(_height, _width * _fused_frames_number, CV_32FC3);
			for (int _i = 0; _i < _fused_frames_number; _i++)
			{
				cv::Mat _frame = _frames[_i];

				for (int c = 0; c < _width; c++) // with each loop take a new line
					_frame.col(c).copyTo(totalframe.col(c * _fused_frames_number + _i));
			}
			Tensor<float>::matrix_to_colored_f_tensor(totalframe, out);
			totalframe = cv::Mat::zeros(_height, _width * _fused_frames_number, _depth);
			if (_draw == 1)
				Tensor<float>::draw_colored_tensor(_file_path + "/Input_frames/" + _expName + "/EF/EF_" + label + "_", out);

			in = out;
		}
	}
	else
	{
		if (_method == 0)
		{ // The output fused tensor
			Tensor<InputType> out(Shape({_height * _fused_frames_number, _width, _depth, _conv_depth / _fused_frames_number}));
			cv::Mat totalframe(_height * _fused_frames_number, _width, CV_32F);
			for (int _i = 0; _i < _fused_frames_number; _i++)
			{
				cv::Mat _frame = _frames[_i];
				for (int r = 0; r < _height; r++) // with each loop take a new line
					_frame.row(r).copyTo(totalframe.row(r * _fused_frames_number + _i));
			}
			Tensor<float>::matrix_to_tensor(totalframe, out);
			totalframe = cv::Mat::zeros(_height * _fused_frames_number, _width, _depth);
			if (_draw == 1)
				Tensor<float>::draw_scaled_tensor(_file_path + "/Input_frames/" + _expName + "/EF/EF_" + label + "_", out, 1);

			in = out;
		}
		else if (_method == 1)
		{ // The output fused tensor
			Tensor<InputType> out(Shape({_height, _width * _fused_frames_number, _depth, _conv_depth / _fused_frames_number}));

			cv::Mat totalframe(_height, _width * _fused_frames_number, CV_32F);
			for (int _i = 0; _i < _fused_frames_number; _i++)
			{
				cv::Mat _frame = _frames[_i];

				for (int c = 0; c < _width; c++) // with each loop take a new line
					_frame.col(c).copyTo(totalframe.col(c * _fused_frames_number + _i));
			}
			Tensor<float>::matrix_to_tensor(totalframe, out);
			totalframe = cv::Mat::zeros(_height, _width * _fused_frames_number, _depth);
			if (_draw == 1)
				Tensor<float>::draw_scaled_tensor(_file_path + "/Input_frames/" + _expName + "/EF/EF_" + label + "_", out, 1);

			in = out;
		}
	}
}