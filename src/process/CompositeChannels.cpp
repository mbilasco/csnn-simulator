#include "process/CompositeChannels.h"

using namespace process;

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

//
//	CompositeChannels
//
static RegisterClassParameter<CompositeChannels, ProcessFactory> _register_1("CompositeChannels");

CompositeChannels::CompositeChannels() : UniquePassProcess(_register_1), _expName(""), _draw(0), _scaler(1), _width(0), _height(0), _depth(0), _conv_depth(0)
{
	add_parameter("draw", _draw);
}

CompositeChannels::CompositeChannels(std::string expName, size_t draw, size_t scaler) : CompositeChannels()
{
	parameter<size_t>("draw").set(draw);
	_expName = expName;
	if (draw == 1)
	{
		std::filesystem::create_directories("Input_frames/" + _expName + "/CC");
		_file_path = std::filesystem::current_path();
	}
}

void CompositeChannels::process_train(const std::string &label, Tensor<float> &sample)
{
	_process(label, sample);
}

void CompositeChannels::process_test(const std::string &label, Tensor<float> &sample)
{
	_process(label, sample);
}

Shape CompositeChannels::compute_shape(const Shape &shape)
{
	_height = shape.dim(0);
	_width = shape.dim(1);
	_depth = 3; // shape.dim(2);
	_conv_depth = shape.dim(3) - 1;
	return Shape({_height, _width, _depth, _conv_depth});
}

void CompositeChannels::_process(const std::string &label, Tensor<InputType> &in) const
{
	std::vector<cv::Mat> _frames;
	std::vector<cv::Mat> _composite_channel_frames;

	Tensor<InputType> out(Shape({_height, _width, _depth, _conv_depth}));

	// This function returns a list of frames that have gone through background subtraction.
	if (in.shape().dim(2) >= 3)
		Tensor<float>::tensor_to_colored_matrices(_frames, in);
	else
		Tensor<float>::tensor_to_matrices(_frames, in);

	cv::Size _frame_size(_width, _height);

	cv::Mat origcopy(_frame_size, CV_8UC3, cv::Scalar(128, 128, 128));

	for (int _i = 0; _i < _frames.size() - 1; _i++)
	{
		// imwrite("/home/melassal/Workspace/CSNN/csnn-simulator-build/Input_frames/test/frame_" + label + "_" + std::to_string(_i) + ".png", _frames[_i]);
		cv::Mat compositeframe(_frame_size, CV_8UC3, cv::Scalar(128, 128, 128));
		cv::Mat img, original, prevgray;

		// capture frames
		prevgray = _frames[_i];
		img = _frames[_i + 1];

		// # ignore if greyscale
		if (_frames[_i].channels() >= 3)
		{
			cv::cvtColor(prevgray, prevgray, cv::COLOR_BGR2GRAY);
			cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
		}

		cv::Mat difference = prevgray - img;
		// To make sure the img is the good size (ignore if good size):
		cv::resize(img, img, _frame_size);
	
		// if previous frame is not empty.. There is a picture of previous frame.
		if (prevgray.empty() == false)
		{
			// calculate optical flow
			cv::Mat flow(_frame_size, CV_32FC1);
			calcOpticalFlowFarneback(prevgray, img, flow, 0.4, 1, 12, 2, 8, 1.2, 0);

			for (int y = 0; y < img.rows; y += 1)
			{
				for (int x = 0; x < img.cols; x += 1)
				{
					// get the flow from y, x position * 10 for better visibility
					const cv::Point2f flowatxy = flow.at<cv::Point2f>(y, x) * 10;
				
					int vx3, vy3;
					vx3 = cvRound(flowatxy.x);
					vy3 = cvRound(flowatxy.y);
					if (vx3 > 127)
					{
						vx3 = 127;
					};
					if (vx3 < -127)
					{
						vx3 = -127;
					};
					if (vy3 > 127)
					{
						vy3 = 127;
					};
					if (vy3 < -127)
					{
						vy3 = -127;
					};
					vx3 = vx3 + 128;
					vy3 = vy3 + 128;

					compositeframe.at<cv::Vec3b>(y, x)[0] = vx3;
					compositeframe.at<cv::Vec3b>(y, x)[1] = vy3;


					int grayscalevalue = img.at<cv::Vec3b>(y, x)[0];
					grayscalevalue += img.at<cv::Vec3b>(y, x)[1];
					grayscalevalue += img.at<cv::Vec3b>(y, x)[2];
					grayscalevalue = grayscalevalue / 3;

					grayscalevalue = difference.at<float>(y, x) != 0 ? grayscalevalue : 0;

					if ((abs(flowatxy.x) > 10) || (abs(flowatxy.y) > 10))
					{
						compositeframe.at<cv::Vec3b>(y, x)[2] = grayscalevalue;
					}
					else
					{
						compositeframe.at<cv::Vec3b>(y, x)[2] = 0;
					};
				}
			}

			img.copyTo(prevgray);
		}
		else
		{
			// fill previous image in case prevgray.empty() == true
			img.copyTo(prevgray);
		}
		_composite_channel_frames.push_back(compositeframe);
	}

	Tensor<float>::matrices_to_colored_tensor(_composite_channel_frames, out);
	if (_draw == 1)
		Tensor<float>::draw_colored_tensor(_file_path + "/Input_frames/" + _expName + "/CC/CC_" + label + "_", out);

	in = out;
}
