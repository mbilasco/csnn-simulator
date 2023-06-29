#include "process/CompositeChannels2.h"

using namespace process;

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

//
//	CompositeChannels2
//
static RegisterClassParameter<CompositeChannels2, ProcessFactory> _register_1("CompositeChannels2");

CompositeChannels2::CompositeChannels2() : UniquePassProcess(_register_1), _expName(""), _draw(0), _scaler(1), _width(0), _height(0), _depth(0), _conv_depth(0)
{
	add_parameter("draw", _draw);
}

CompositeChannels2::CompositeChannels2(std::string expName, size_t draw, size_t scaler) : CompositeChannels2()
{
	parameter<size_t>("draw").set(draw);
	_expName = expName;
	if (draw == 1)
	{
		std::filesystem::create_directories("Input_frames/" + _expName + "/CC2");
		_file_path = std::filesystem::current_path();
	}
}

void CompositeChannels2::process_train(const std::string &label, Tensor<float> &sample)
{
	_process(label, sample);
}

void CompositeChannels2::process_test(const std::string &label, Tensor<float> &sample)
{
	_process(label, sample);
}

Shape CompositeChannels2::compute_shape(const Shape &shape)
{
	_height = shape.dim(0);
	_width = shape.dim(1);
	_depth = 3; // shape.dim(2);
	_conv_depth = shape.dim(3) - 1;
	return Shape({_height, _width, _depth, _conv_depth});
}

void CompositeChannels2::_process(const std::string &label, Tensor<InputType> &in) const
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

	cv::Mat oldvelocityx(_frame_size, CV_8UC3, cv::Scalar(128, 128, 128));
	cv::Mat oldvelocityy(_frame_size, CV_8UC3, cv::Scalar(128, 128, 128));
	cv::Mat velocityx(_frame_size, CV_8UC3, cv::Scalar(128, 128, 128));
	cv::Mat velocityy(_frame_size, CV_8UC3, cv::Scalar(128, 128, 128));

	cv::Mat origcopy(_frame_size, CV_8UC3, cv::Scalar(128, 128, 128));

	for (int _i = 0; _i < _frames.size() - 1; _i++)
	{
		// imwrite("/home/melassal/Workspace/CSNN/csnn-simulator-build/Input_frames/test/frame_" + label + "_" + std::to_string(_i) + ".png", _frames[_i]);
		cv::Mat compositeframe(_frame_size, CV_8UC3, cv::Scalar(128, 128, 128));
		cv::Mat img, original, prevgray;

		// capture frames
		prevgray = _frames[_i];
		img = _frames[_i + 1];

		if (_frames[_i].channels() >= 3)
		{
			cv::cvtColor(prevgray, prevgray, cv::COLOR_BGR2GRAY);
			cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
		}
		// To flip a video use this command:
		cv::resize(img, img, _frame_size);
		// save original for later
		img.copyTo(original);
		img.copyTo(origcopy);

		// optical flow
		cv::Mat flow(_frame_size, CV_32FC1);
		cv::calcOpticalFlowFarneback(prevgray, img, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
		// visualization
		cv::Mat flow_parts[2];
		split(flow, flow_parts);
		cv::Mat magnitude, angle, magn_norm;
		// changinf the coordinates to polar in order to use them in the hsv representation.
		cv::cartToPolar(flow_parts[0] * _scaler, flow_parts[1] * _scaler, magnitude, angle, true);
		cv::normalize(magnitude, magn_norm, 0.0f, 1.0f, cv::NORM_MINMAX);
		angle *= ((1.f / 360.f) * (180.f / 255.f));
		// build hsv image
		cv::Mat _hsv[3], hsv, hsv8, bgr;
		_hsv[0] = angle;
		_hsv[1] = cv::Mat::ones(angle.size(), CV_32F);
		_hsv[2] = magn_norm;
		merge(_hsv, 3, hsv);
		hsv.convertTo(hsv8, CV_8U, 255.0);
		// convert the HSV image into RGB.
		cvtColor(hsv8, bgr, cv::COLOR_HSV2BGR);

		// if previous frame is not empty.. There is a picture of previous frame.
		if (prevgray.empty() == false)
		{
			// calculate optical flow
			cv::Mat flow(_frame_size, CV_32FC1);
			calcOpticalFlowFarneback(prevgray, img, flow, 0.4, 1, 12, 2, 8, 1.2, 0);

			// By y += 5, x += 5 you can specify the grid
			velocityx.copyTo(oldvelocityx);
			velocityy.copyTo(oldvelocityy);

			for (int y = 0; y < original.rows; y += 1)
			{
				for (int x = 0; x < original.cols; x += 1)
				{
					// get the flow from y, x position * 10 for better visibility
					const cv::Point2f flowatxy = flow.at<cv::Point2f>(y, x) * 10;
					// draw line at flow direction
					int x2, y2;

					x2 = cvRound(x + flowatxy.x);
					y2 = cvRound(y + flowatxy.y);

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

					if ((x % 10 == 0) && (y % 10 == 0))
						line(original, cv::Point(x, y), cv::Point(x2, y2), cv::Scalar(255, 0, 0));

					velocityx.at<cv::Vec3b>(y, x)[0] = vx3;
					velocityx.at<cv::Vec3b>(y, x)[1] = vx3;
					velocityx.at<cv::Vec3b>(y, x)[2] = vx3;
					velocityy.at<cv::Vec3b>(y, x)[0] = vy3;
					velocityy.at<cv::Vec3b>(y, x)[1] = vy3;
					velocityy.at<cv::Vec3b>(y, x)[2] = vy3;

					int optflowvalue = bgr.at<cv::Vec3b>(y, x)[0];
					optflowvalue += bgr.at<cv::Vec3b>(y, x)[1];
					optflowvalue += bgr.at<cv::Vec3b>(y, x)[2];
					optflowvalue = optflowvalue / 3;
					if (optflowvalue > 255)
					{
						optflowvalue = 255;
					};

					compositeframe.at<cv::Vec3b>(y, x)[0] = vx3;
					compositeframe.at<cv::Vec3b>(y, x)[1] = vy3;

					if ((abs(flowatxy.x) > 10) || (abs(flowatxy.y) > 10))
					{
						compositeframe.at<cv::Vec3b>(y, x)[2] = optflowvalue;
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
		Tensor<float>::draw_colored_tensor(_file_path + "/Input_frames/" + _expName + "/CC2/CC2_" + label + "_", out);

	in = out;
}
