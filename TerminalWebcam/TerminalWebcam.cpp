#include <string_view>
#include <map>

#ifdef _MSC_VER // These #pragma lines are MSVC-specific!
#pragma warning(push)
#pragma warning(disable:26495)  //
#pragma warning(disable:6201)   // Disable specified warning numbers
#pragma warning(disable:4365)   // 
#pragma warning(disable:6294)   //
#endif // _MSC_VER


#include "curses.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/photo.hpp"

#ifdef _MSC_VER
#pragma warning(pop)
#endif // _MSC_VER

// https://tldp.org/HOWTO/NCURSES-Programming-HOWTO/

uchar ucharToGradient(uchar, uchar);
void colorReduce(cv::Mat&, uchar);

// #define SHOW_DEBUG_WINDOW
#define SHOW_WEBCAM

// #define COLOR_MODE // Experimental and very slow

/* CHOOSE ONE FILTER - Filter1 has the most character whereas filter5 has only 1
	Recommended filter is FILTER3
*/
// #define FILTER1
// #define FILTER2
 #define FILTER3
// #define FILTER4
// #define FILTER5
// #define FILTER3_INVERTED

int main()
{
	// Setup NCurses
	initscr();
	raw();
	keypad(stdscr, TRUE);
#ifdef COLOR_MODE
	start_color();
#endif // COLOR_MODE


	// Good for 4:3 ratio,
	// Width is around x2 of original image because -
	// of how tall/thin chars are in the terminal
#ifndef COLOR_MODE
	resize_term(120, 320);
#endif // !COLOR_MODE

#ifdef COLOR_MODE
	resize_term(120 / 4, 320 / 4);
#endif // COLOR_MODE


	// Setup Video capture
	cv::VideoCapture cap = cv::VideoCapture(0);
	assert(cap.isOpened() && "Could not find or open a camera");

	// The "image" we will be referencing
	cv::Mat frame;
	cv::Mat grayframe;

	constexpr uchar colorReductionAmount{ 32 };
	constexpr uchar lowLightLevelFilter{ 60 };

#ifdef COLOR_MODE
	std::map<std::pair<int, int>, WINDOW*> windows;

	for (int y = 0; y < stdscr->_maxy; y++)
	{
		for (int x = 0; x < stdscr->_maxx; x++)
		{
			windows.insert(std::make_pair(std::make_pair(y, x), newwin(1, 1, y, x)));
		}
	}
#endif // COLOR_MODE


	// Press ESC in OpenCV window to abort
	while (cv::waitKey(1) != 27)
	{
		// Get frame from video capture
		cap.read(frame);
#ifdef SHOW_WEBCAM
		cv::imshow("WEBCAM", frame);
#endif // SHOW_WEBCAM

		// Resize frame to fit terminal
		cv::resize(frame, frame, cv::Size(stdscr->_maxx, stdscr->_maxy), 0, 0, cv::INTER_NEAREST);

		colorReduce(frame, colorReductionAmount);

#ifndef COLOR_MODE
		// Convert to grayscale
		cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
		cv::fastNlMeansDenoising(frame, frame, 7, 7, 7);
#endif // !COLOR_MODE

#ifdef COLOR_MODE
		cv::cvtColor(frame, grayframe, cv::COLOR_BGR2GRAY);
		cv::fastNlMeansDenoising(grayframe, grayframe, 21, 7, 7);
#endif // COLOR_MODE


#ifdef SHOW_DEBUG_WINDOW
		cv::imshow("DEBUG WINDOW: Frame after Post-Processing", frame);
#endif // SHOW_DEBUG_WINDOW

#ifndef COLOR_MODE
		// Draw frame to console using NCurses
		for (int y = 0; y < frame.rows; y++)
		{
			for (int x = 0; x < frame.cols; x++)
			{
				uchar pixel = frame.at<uchar>(y, x);
				uchar charToDraw = ucharToGradient(pixel, lowLightLevelFilter);
				mvaddch(y, x, charToDraw);
			}
		}
#endif // !COLOR_MODE

#ifdef COLOR_MODE
		std::map<std::tuple<uchar, uchar, uchar>, short> colorPairs{};

		// Draw frame to console using NCurses
		for (int y = 0; y < frame.rows; y++)
		{
			for (int x = 0; x < frame.cols; x++)
			{
				WINDOW* window = windows.at(std::make_pair(y, x));

				uchar charToPrint{ ucharToGradient(grayframe.at<uchar>(y, x), lowLightLevelFilter) };
				/* This experimental optimization gives wrong color info - may work with a stable and not noisy camera
				chtype charCurrentlyPrinted{ mvwinch(window, 0, 0) & A_CHARTEXT };
				if (charToPrint == charCurrentlyPrinted)
				{
					continue;
				}
				*/
			
				mvwaddch(window, 0, 0, charToPrint);

				// If char is not blank then we need to color it
				if (charToPrint != ' ')
				{
					cv::Vec3b bgrPixel = frame.at<cv::Vec3b>(y, x);
					std::tuple bgrKey = std::make_tuple(bgrPixel[0], bgrPixel[1], bgrPixel[2]);

					if (!colorPairs.contains(bgrKey))
					{
						short colorPairSize{ (short)(colorPairs.size() + 7) };
						short nextIndex{ colorPairSize >= SHRT_MAX ? SHRT_MAX : colorPairSize + 1 };
						colorPairs[bgrKey] = nextIndex;
						init_color(nextIndex, 
							bgrPixel[2] * 1000 / UCHAR_MAX,
							bgrPixel[1] * 1000 / UCHAR_MAX,
							bgrPixel[0] * 1000 / UCHAR_MAX);
						init_pair(nextIndex, nextIndex, COLOR_BLACK);
						wattrset(window, COLOR_PAIR(nextIndex));
					}
					else {
						short color = colorPairs[bgrKey];
						init_pair(color, color, COLOR_BLACK);
						wattrset(window, COLOR_PAIR(color));
					}
				}

				wrefresh(window);
			}
		}
#endif // COLOR_MODE

#ifndef COLOR_MODE
		// After finish drawing, refresh to show frame in console
		refresh();
#endif // !COLOR_MODE
	}

	// Release video capture and end ncurses
	cap.release();
	endwin();
	return 0;
}

// lowLightFilter usually a value between 0-255 (uchar), filters low level areas as blanks instead of dots
uchar ucharToGradient(uchar ch, uchar lowLightFilterLevel) {
	if (ch <= lowLightFilterLevel)
	{
		return ' ';
	}

#ifdef FILTER1
	static constexpr std::string_view gradient = R"(.'`^",:;Il!i><~+_-?][}{1)(|\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$)";
#endif // FILTER1

#ifdef FILTER2
	static constexpr std::string_view gradient = R"(.^":-=+*?O#%@$)";
#endif // FILTER2

#ifdef FILTER3
	static constexpr std::string_view gradient = R"(.:-=+*#%@)";
#endif // FILTER3

#ifdef FILTER4
	static constexpr std::string_view gradient = R"(#%@)";
#endif // FILTER4

#ifdef FILTER5
	static constexpr std::string_view gradient = R"(#%@)";
	return '@';
#endif // FILTER4

#ifdef FILTER3_INVERTED
	static constexpr std::string_view gradient = R"(@%#*+=-:.)";
#endif // FILTER3_INVERTED


	static constexpr size_t gradientMax = gradient.length() - 1;
	const size_t index = (size_t)(ch / (double)((double)255 / gradientMax));

	return (uchar)gradient.at(index);
}

void colorReduce(cv::Mat& image, uchar div)
{
	int nl = image.rows;                    // number of lines
	int nc = image.cols * image.channels(); // number of elements per line

	for (int j = 0; j < nl; j++)
	{
		// get the address of row j
		uchar* data = image.ptr<uchar>(j);

		for (int i = 0; i < nc; i++)
		{
			// process each pixel
			data[i] = (uchar)(data[i] / div * div + div / 2);
		}
	}
}
