/*
 * Name: Yanting Lai
 * Date: September 22, 2024
 * Purpose:
 * This program captures live video from a specified file and displays it in a window. 
 * The user can interactively apply various image filters using specific keypresses:
 * - 'g': OpenCV grayscale
 * - 'h': Custom grayscale
 * - 'p': Naive 5x5 blur
 * - 'b': Optimized 5x5 blur
 * - 'x': Sobel X edge detection
 * - 'y': Sobel Y edge detection
 * - 'm': Gradient magnitude (combining Sobel X and Y)
 * - 'l': Blur and quantize
 * - 'f': Face detection
 * - 'e': Emboss effect
 * - 'c': Cartoonization
 * - 'n': Negative filter
 * - 'k': Median filter
 *
 * Additional functionality includes:
 * - 'q': Quit the program
 * - 's': Save the current frame as an image.
 *
 * The program also prints the execution time of certain filters to the console.
 * 
 */

#include <opencv2/opencv.hpp>
#include "faceDetect.h"
#include "filters.hpp"  // Include the custom filter header
#include <iostream>

enum FilterType {
    NO_FILTER,
    GRAY,
    CUSTOM_GRAY,
    BLUR1,
    BLUR2,
    SOBEL_X,
    SOBEL_Y,
    MAGNITUDE,
    BLUR_QUANTIZE,
    FACE_DETECT,
    EMBOSS,
    CARTOONIZE,
    NEGATIVE,
    MEDIAN
};


int main() {
    // Open the video file
    cv::VideoCapture capdev("/Users/sundri/Desktop/CS5330/Project/Project1/CPA_Intro_Yanting.mp4");

    if (!capdev.isOpened()) {
        std::cerr << "Error: Could not open the video file!" << std::endl;
        return -1;
    }

    // Get some properties of the video
    cv::Size refS((int) capdev.get(cv::CAP_PROP_FRAME_WIDTH), (int) capdev.get(cv::CAP_PROP_FRAME_HEIGHT));
    std::cout << "Expected size: " << refS.width << " x " << refS.height << std::endl;

    cv::namedWindow("Video", cv::WINDOW_AUTOSIZE);  // Create a window for video display
    std::vector<cv::Rect> faces;  // To store detected faces
    cv::Mat frame, grayFrame, customGrayFrame, blurFrame1, blurFrame2, sobelXFrame, sobelYFrame, absSobelXFrame, absSobelYFrame, blurQuantizeFrame, absMagnitudeFrame, dst, grey, cartoonFrame,medianFrame;
    int frameCount = 0;
    FilterType currentFilter = NO_FILTER;

    std::string original_frame, filtered_frame;

    // Loop to capture and display video frames
    while (true) {
        capdev >> frame;  // Get a new frame from the video file
        if (frame.empty()) {
            std::cout << "End of video reached or frame is empty." << std::endl;
            break;
        }

        double start, end; // Variables to hold timing data

        // Apply the current filter based on the value of currentFilter
        switch (currentFilter) {
            case GRAY:
                if (frame.channels() == 3) {
                    original_frame = "before_gray_filter" + std::to_string(frameCount) + ".jpg";
                    cv::imwrite(original_frame, frame);
                    cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
                    cv::imshow("Video", grayFrame);
                    filtered_frame = "after_gray_filter" + std::to_string(frameCount) + ".jpg";
                    cv::imwrite(filtered_frame, grayFrame);
                } else {
                    std::cerr << "Warning: Frame is not a 3-channel image!" << std::endl;
                }
                break;
            case CUSTOM_GRAY:
                original_frame = "before_custom_gray_filter" + std::to_string(frameCount) + ".jpg";
                cv::imwrite(original_frame, frame);
                greyscale(frame, customGrayFrame);
                cv::imshow("Video", customGrayFrame);
                filtered_frame = "after_custom_gray_filter" + std::to_string(frameCount) + ".jpg";
                cv::imwrite(filtered_frame, customGrayFrame);
                break;
            case BLUR1:
                start = cv::getTickCount();  // Start timing
                blur5x5_1(frame, blurFrame1);
                end = cv::getTickCount();  // End timing
                std::cout << "blur5x5_1 Time: " << (end - start) / cv::getTickFrequency() << " seconds" << std::endl;
                cv::imshow("Video", blurFrame1);
                break;
            case BLUR2:
                start = cv::getTickCount();  // Start timing
                blur5x5_2(frame, blurFrame2);
                end = cv::getTickCount();  // End timing
                std::cout << "blur5x5_2 Time: " << (end - start) / cv::getTickFrequency() << " seconds" << std::endl;
                cv::imshow("Video", blurFrame2);
                break;
            case SOBEL_X:
                original_frame = "before_sobel_x_filter" + std::to_string(frameCount) + ".jpg";
                cv::imwrite(original_frame, frame);
                sobelX3x3(frame, sobelXFrame);
                cv::convertScaleAbs(sobelXFrame, absSobelXFrame);
                cv::imshow("Video", absSobelXFrame);
                filtered_frame = "after_sobel_x_filter" + std::to_string(frameCount) + ".jpg";
                cv::imwrite(filtered_frame, absSobelXFrame);
                break;
            case SOBEL_Y:
                original_frame = "before_sobel_y_filter" + std::to_string(frameCount) + ".jpg";
                cv::imwrite(original_frame, frame);
                sobelY3x3(frame, sobelYFrame);
                cv::convertScaleAbs(sobelYFrame, absSobelYFrame);
                cv::imshow("Video", absSobelYFrame);
                filtered_frame = "after_sobel_y_filter" + std::to_string(frameCount) + ".jpg";
                cv::imwrite(filtered_frame, absSobelYFrame);
                break;
            case MAGNITUDE:
                original_frame = "before_magnitude_filter" + std::to_string(frameCount) + ".jpg";
                cv::imwrite(original_frame, frame);
                sobelX3x3(frame, sobelXFrame);
                sobelY3x3(frame, sobelYFrame);
                magnitude(sobelXFrame, sobelYFrame, absMagnitudeFrame);
                cv::imshow("Video", absMagnitudeFrame);
                filtered_frame = "after_magnitude_filter" + std::to_string(frameCount) + ".jpg";
                cv::imwrite(filtered_frame, absMagnitudeFrame);
                break;
            case BLUR_QUANTIZE:
                original_frame = "before_blur_quantize_filter" + std::to_string(frameCount) + ".jpg";
                cv::imwrite(original_frame, frame);
                blurQuantize(frame, blurQuantizeFrame, 10);  // Quantize with 10 levels
                cv::imshow("Video", blurQuantizeFrame);
                filtered_frame = "after_blur_quantize_filter" + std::to_string(frameCount) + ".jpg";
                cv::imwrite(filtered_frame, blurQuantizeFrame);
                break;
            case FACE_DETECT:
                original_frame = "before_face_detect" + std::to_string(frameCount) + ".jpg";
                cv::imwrite(original_frame, frame);
                cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);  // Convert frame to greyscale
                detectFaces(grey, faces);
                drawBoxes(frame, faces);
                cv::imshow("Video", frame);
                filtered_frame = "after_face_detect" + std::to_string(frameCount) + ".jpg";
                cv::imwrite(filtered_frame, frame);
                break;
            case EMBOSS:
                original_frame = "before_emboss_detect" + std::to_string(frameCount) + ".jpg";
                cv::imwrite(original_frame, frame);
                emboss(frame, dst);
                cv::imshow("Video", dst);
                filtered_frame = "after_emboss_detect" + std::to_string(frameCount) + ".jpg";
                cv::imwrite(filtered_frame, dst);
                break;
            case CARTOONIZE:
                cartoonize(frame, cartoonFrame);
                cv::imshow("Video", cartoonFrame);
                break;
            case NEGATIVE:
                original_frame = "before_negative" + std::to_string(frameCount) + ".jpg";
                cv::imwrite(original_frame, frame);
                negative(frame, dst);
                cv::imshow("Video", dst);  // Show the negative image
                filtered_frame = "after_negative" + std::to_string(frameCount) + ".jpg";
                cv::imwrite(filtered_frame, dst);
                break;
            case MEDIAN:
                original_frame = "before_medium" + std::to_string(frameCount) + ".jpg";
                cv::imwrite(original_frame, frame);
                cv::medianBlur(frame, medianFrame, 7);
                cv::imshow("Video", medianFrame);
                filtered_frame = "after_medium" + std::to_string(frameCount) + ".jpg";
                cv::imwrite(filtered_frame, medianFrame);
                break;
            case NO_FILTER:
            default:
                cv::imshow("Video", frame);  // Display the original frame if no filter
                break;
        }

        // Check for a keypress
        char key = cv::waitKey(10);
        if (key == 'q') {
            break;
        } else if (key == 's') {
            // Save the current frame to a file
            std::string filename = "saved_frame_" + std::to_string(frameCount) + ".jpg";
            cv::imwrite(filename, frame);
            std::cout << "Saved frame to " << filename << std::endl;
            frameCount++;
        } else if (key == 'g') {
            currentFilter = (currentFilter == GRAY) ? NO_FILTER : GRAY;
        } else if (key == 'h') {
            currentFilter = (currentFilter == CUSTOM_GRAY) ? NO_FILTER : CUSTOM_GRAY;
        } else if (key == 'p') {
            currentFilter = (currentFilter == BLUR1) ? NO_FILTER : BLUR1;
        } else if (key == 'b') {
            currentFilter = (currentFilter == BLUR2) ? NO_FILTER : BLUR2;
        } else if (key == 'x') {
            currentFilter = (currentFilter == SOBEL_X) ? NO_FILTER : SOBEL_X;
        } else if (key == 'y') {
            currentFilter = (currentFilter == SOBEL_Y) ? NO_FILTER : SOBEL_Y;
        } else if (key == 'm') {
            currentFilter = (currentFilter == MAGNITUDE) ? NO_FILTER : MAGNITUDE;
        } else if (key == 'l') {
            currentFilter = (currentFilter == BLUR_QUANTIZE) ? NO_FILTER : BLUR_QUANTIZE;
        } else if (key == 'f') {
            currentFilter = (currentFilter == FACE_DETECT) ? NO_FILTER : FACE_DETECT;
        } else if (key == 'e') {
            currentFilter = (currentFilter == EMBOSS) ? NO_FILTER : EMBOSS;
        } else if (key == 'c') {
            currentFilter = (currentFilter == CARTOONIZE) ? NO_FILTER : CARTOONIZE;
        } else if (key == 'n') {
            currentFilter = (currentFilter == NEGATIVE) ? NO_FILTER : NEGATIVE;
        } else if (key == 'k') {
            currentFilter = (currentFilter == MEDIAN) ? NO_FILTER : MEDIAN;
        }
    }

    return 0;
}
