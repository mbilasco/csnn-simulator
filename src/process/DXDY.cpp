#include "process/DXDY.h"

using namespace process;

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

//
//	DXDY
//
static RegisterClassParameter<DXDY, ProcessFactory> _register_1("DXDY");

DXDY::DXDY() : UniquePassProcess(_register_1), _expName(""), _draw(0), _scaler(1), _width(0), _height(0), _depth(0), _conv_depth(0)
{
	add_parameter("draw", _draw);
	add_parameter("scaler", _scaler);
}

DXDY::DXDY(std::string expName, size_t draw, size_t scaler) : DXDY()
{
	parameter<size_t>("draw").set(draw);
	parameter<size_t>("scaler").set(scaler);
	_expName = expName;
	if (draw == 1)
	{
		std::filesystem::create_directories("Input_frames/" + _expName + "/DXDY");
	}
	_file_path = std::filesystem::current_path();
}

void DXDY::process_train(const std::string &label, Tensor<float> &sample)
{
	_process(label, sample);
}

void DXDY::process_test(const std::string &label, Tensor<float> &sample)
{
	_process(label, sample);
}

Shape DXDY::compute_shape(const Shape &shape)
{
	_height = shape.dim(0);
	_width = shape.dim(1);
	_depth = 3; // shape.dim(2);
	_conv_depth = shape.dim(3) - 1;
	return Shape({_height, _width, _depth, _conv_depth});
}

void DXDY::_process(const std::string &label, Tensor<InputType> &in) const
{
	std::vector<cv::Mat> _frames;
	std::vector<cv::Mat> _out_frames;

	Tensor<InputType> out(Shape({_height, _width, _depth, _conv_depth}));

	// This function returns a list of frames that have gone through background subtraction.
	if (in.shape().dim(2) >= 3)
		Tensor<float>::tensor_to_colored_matrices(_frames, in);
	else
		Tensor<float>::tensor_to_matrices(_frames, in);

	float SCALER = _scaler;

	cv::Size _frame_size(_width, _height);

	for (int _i = 0; _i < _frames.size() - 1; _i++)
	{
		// imwrite("/home/melassal/Workspace/CSNN/csnn-simulator-build/Input_frames/test/frame_" + label + "_" + std::to_string(_i) + ".png", _frames[_i]);
		cv::resize(_frames[_i], _frames[_i], _frame_size);
		cv::resize(_frames[_i + 1], _frames[_i + 1], _frame_size);

		cv::Mat _frame1 = _frames[_i];
		cv::Mat _frame2 = _frames[_i + 1];

		if (_frame1.channels() > 1)
		{
			cv::cvtColor(_frame1, _frame1, cv::COLOR_BGR2GRAY);
			cv::cvtColor(_frame2, _frame2, cv::COLOR_BGR2GRAY);
		}
		cv::Mat flow(_frame_size, CV_32FC1);
		cv::calcOpticalFlowFarneback(_frame1, _frame2, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

		cv::Mat flow_parts[2];
		split(flow, flow_parts);

		// adding the flow parts in separate channels.
		cv::Mat _frame(_height, _width, CV_32FC3);

		for (size_t i = 0; i < _height; i++)
			for (size_t j = 0; j < _width; j++)
			{
				_frame.at<cv::Vec3f>(i, j)[0] = flow_parts[0].at<float>(i, j) * _scaler;
				_frame.at<cv::Vec3f>(i, j)[1] = flow_parts[1].at<float>(i, j) * _scaler;
				_frame.at<cv::Vec3f>(i, j)[2] = 0;
			}
		_out_frames.push_back(_frame);
	}
	Tensor<float>::matrices_to_colored_f_tensor(_out_frames, out);
	if (_draw == 1)
		Tensor<float>::draw_scaled_colored_tensor(_file_path + "/Input_frames/" + _expName + "/DXDY/DXDY_" + label + "_", out);

	in = out;
}