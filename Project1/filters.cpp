/*
 * Yanting Lai
 * Date: 2024-09-22
 * CS 5330 - Computer Vision
 * Purpose: Detail function implementations of filters.hpp
 */

#include "filters.hpp"
#include <opencv2/opencv.hpp>

// custom greyscale transformation
int greyscale(cv::Mat &src, cv::Mat &dst) {
    // Check if source image is empty
    if (src.empty()) {
        return -1;
    }

    // Create destination matrix with the same size and type as source
    dst.create(src.size(), src.type());

    // Loop through each pixel to apply custom greyscale transformation
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);

            // Custom greyscale transformation:
            uchar greyValue = 255 - pixel[2];  // Reverse red channel

            // Assign greyValue to all three channels (B, G, R)
            dst.at<cv::Vec3b>(i, j)[0] = greyValue;  // Blue
            dst.at<cv::Vec3b>(i, j)[1] = greyValue;  // Green
            dst.at<cv::Vec3b>(i, j)[2] = greyValue;  // Red
        }
    }

    return 0;
}

// sepia filter
int sepia(cv::Mat &src, cv::Mat &dst) {
    // Check if source image is empty
    if (src.empty()) {
        return -1;
    }

    // Create destination matrix with the same size and type as source
    dst.create(src.size(), src.type());

    // Calculate the image center
    int centerX = src.cols / 2;
    int centerY = src.rows / 2;
    // Calculate the maximum possible distance from the center
    double maxDistance = std::sqrt(centerX * centerX + centerY * centerY);

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            // Get the original pixel (BGR format)
            cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);

            uchar old_blue = pixel[0];
            uchar old_green = pixel[1];
            uchar old_red = pixel[2];

            // Apply the sepia tone transformation matrix
            uchar new_blue = std::min(255.0, 0.272 * old_red + 0.534 * old_green + 0.131 * old_blue);
            uchar new_green = std::min(255.0, 0.349 * old_red + 0.686 * old_green + 0.168 * old_blue);
            uchar new_red = std::min(255.0, 0.393 * old_red + 0.769 * old_green + 0.189 * old_blue);

            // Calculate the distance of the pixel from the center of the image
            double distance = std::sqrt((j - centerX) * (j - centerX) + (i - centerY) * (i - centerY));

            // Compute the vignette factor (a value between 0.0 and 1.0 based on the distance)
            double vignetteFactor = 1.0 - (distance / maxDistance);
            vignetteFactor = std::pow(vignetteFactor, 2.0);  // Optional: adjust how strong the vignetting effect is

            // Apply vignetting by scaling the new color values based on the vignette factor
            new_blue = static_cast<uchar>(new_blue * vignetteFactor);
            new_green = static_cast<uchar>(new_green * vignetteFactor);
            new_red = static_cast<uchar>(new_red * vignetteFactor);

            // Assign new values to the destination image
            dst.at<cv::Vec3b>(i, j)[0] = new_blue;
            dst.at<cv::Vec3b>(i, j)[1] = new_green;
            dst.at<cv::Vec3b>(i, j)[2] = new_red;
        }
    }

    return 0;
}

// blur filter version one for vidDisplay.cpp
int blur5x5_1(cv::Mat &src, cv::Mat &dst) 
{
    if (src.empty()) {
        return -1;
    }

    // Create the destination image with the same size and type as the source image
    dst.create(src.size(), src.type());

    // Define the 5x5 kernel
    int kernel[5][5] = {
        {1, 2, 4, 2, 1},
        {2, 4, 8, 4, 2},
        {4, 8, 16, 8, 4},
        {2, 4, 8, 4, 2},
        {1, 2, 4, 2, 1}
    };
    
    int kernel_sum = 100;  // Sum of all kernel values

    // Loop over each pixel in the image (excluding the outer two rows/columns)
    for (int y = 2; y < src.rows - 2; y++) {
        for (int x = 2; x < src.cols - 2; x++) {
            // Process each color channel separately
            for (int c = 0; c < 3; c++) {
                int sum = 0;

                // Apply the 5x5 kernel to the surrounding pixels
                for (int ky = -2; ky <= 2; ky++) {
                    for (int kx = -2; kx <= 2; kx++) {
                        sum += src.at<cv::Vec3b>(y + ky, x + kx)[c] * kernel[ky + 2][kx + 2];
                    }
                }

                // Normalize the result and assign it to the destination image
                dst.at<cv::Vec3b>(y, x)[c] = cv::saturate_cast<uchar>(sum / kernel_sum);
            }
        }
    }

    return 0;
}

// optmized blur filter
int blur5x5_2(cv::Mat &src, cv::Mat &dst) 
{
    if (src.empty()) {
        return -1;  // Check for an empty image
    }

    dst.create(src.size(), src.type());  // Create the destination image

    // Kernel for the 1x5 filter
    int kernel[5] = {1, 2, 4, 2, 1};
    int kernel_sum = 10;

    // Temporary image for the horizontal pass
    cv::Mat temp = cv::Mat::zeros(src.size(), src.type());

    // Horizontal pass
    for (int y = 0; y < src.rows; y++) {
        for (int x = 2; x < src.cols - 2; x++) {
            for (int c = 0; c < 3; c++) {  // Process each color channel separately
                int sum = 0;
                for (int k = -2; k <= 2; k++) {
                    sum += src.at<cv::Vec3b>(y, x + k)[c] * kernel[k + 2];
                }
                temp.at<cv::Vec3b>(y, x)[c] = sum / kernel_sum;  // Normalize
            }
        }
    }

    // Vertical pass
    for (int y = 2; y < src.rows - 2; y++) {
        for (int x = 0; x < src.cols; x++) {
            for (int c = 0; c < 3; c++) {
                int sum = 0;
                for (int k = -2; k <= 2; k++) {
                    sum += temp.at<cv::Vec3b>(y + k, x)[c] * kernel[k + 2];
                }
                dst.at<cv::Vec3b>(y, x)[c] = sum / kernel_sum;  // Normalize
            }
        }
    }

    return 0;
}

// Sobel X 3x3 filter
int sobelX3x3(const cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        return -1;
    }

    // Create destination image with the same size and 16-bit signed type
    dst.create(src.size(), CV_16SC3);

    // Sobel X filter kernel
    int kernelX[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    // Loop over each pixel (excluding the outermost rows/columns)
    for (int y = 1; y < src.rows - 1; y++) {
        for (int x = 1; x < src.cols - 1; x++) {
            cv::Vec3s color(0, 0, 0);

            // Apply the 3x3 Sobel X filter
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    cv::Vec3b pixel = src.at<cv::Vec3b>(y + ky, x + kx);  // Access the source pixel
                    int kernel_value = kernelX[ky + 1][kx + 1];

                    // Accumulate the Sobel X response for each color channel
                    color[0] += pixel[0] * kernel_value;  // Blue
                    color[1] += pixel[1] * kernel_value;  // Green
                    color[2] += pixel[2] * kernel_value;  // Red
                }
            }

            dst.at<cv::Vec3s>(y, x) = color;  // Store the result in the destination image
        }
    }

    return 0;
}

// Sobel Y 3x3 filter
int sobelY3x3(const cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        return -1;
    }

    // Create destination image with the same size and 16-bit signed type
    dst.create(src.size(), CV_16SC3);

    // Sobel Y filter kernel
    int kernelY[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    // Loop over each pixel (excluding the outermost rows/columns)
    for (int y = 1; y < src.rows - 1; y++) {
        for (int x = 1; x < src.cols - 1; x++) {
            cv::Vec3s color(0, 0, 0);  // 16-bit signed vector for each color channel

            // Apply the 3x3 Sobel Y filter
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    cv::Vec3b pixel = src.at<cv::Vec3b>(y + ky, x + kx);  // Access the source pixel
                    int kernel_value = kernelY[ky + 1][kx + 1];

                    // Accumulate the Sobel Y response for each color channel
                    color[0] += pixel[0] * kernel_value;  // Blue
                    color[1] += pixel[1] * kernel_value;  // Green
                    color[2] += pixel[2] * kernel_value;  // Red
                }
            }

            dst.at<cv::Vec3s>(y, x) = color;  // Store the result in the destination image
        }
    }

    return 0;
}

// compute gradient magnitude from Sobel X and Y images
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst) {
    if (sx.empty() || sy.empty()) {
        return -1;  // Check if input images are empty
    }

    // Create destination image with the same size, but 8-bit unsigned type
    dst.create(sx.size(), CV_8UC3);

    // Loop through each pixel to compute the gradient magnitude
    for (int y = 0; y < sx.rows; y++) {
        for (int x = 0; x < sx.cols; x++) {
            cv::Vec3s pixelX = sx.at<cv::Vec3s>(y, x);  // Sobel X pixel
            cv::Vec3s pixelY = sy.at<cv::Vec3s>(y, x);  // Sobel Y pixel
            cv::Vec3b magnitudePixel;  // Destination pixel

            // Calculate magnitude for each color channel (B, G, R)
            for (int c = 0; c < 3; c++) {
                int mag = std::sqrt(pixelX[c] * pixelX[c] + pixelY[c] * pixelY[c]);  // Euclidean distance
                magnitudePixel[c] = cv::saturate_cast<uchar>(mag);  // Ensure value is within [0, 255]
            }

            dst.at<cv::Vec3b>(y, x) = magnitudePixel;  // Assign the magnitude to the destination image
        }
    }

    return 0;
}

// blur and quantize a color image
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels) {
    if (src.empty()) {
        return -1;
    }

    // Step 1: Apply a blur filter
    blur5x5_1(src, dst);  // First blur the image using the previously implemented function

    // Step 2: Quantize the image
    int b = 255 / levels;

    // Loop through each pixel and quantize the color values
    for (int y = 0; y < dst.rows; y++) {
        for (int x = 0; x < dst.cols; x++) {
            cv::Vec3b &pixel = dst.at<cv::Vec3b>(y, x);

            // Quantize each color channel (B, G, R)
            for (int c = 0; c < 3; c++) {
                int xt = pixel[c] / b;  // Compute the quantized bucket index
                pixel[c] = xt * b;  // Set the quantized value
            }
        }
    }

    return 0;
}

// Uses the Sobel X and Y filters to create an emboss effect
void emboss(const cv::Mat &src, cv::Mat &dst) {
    cv::Mat nonConstSrc = src.clone();  // Create a non-const copy of src
    cv::Mat sobelX, sobelY;

    // Apply Sobel filters
    sobelX3x3(nonConstSrc, sobelX);
    sobelY3x3(nonConstSrc, sobelY);

    // Combine Sobel X and Y to create the emboss effect
    dst = 0.5 * sobelX + 0.5 * sobelY + 128;  // 128 shifts the result into the valid range

    cv::convertScaleAbs(dst, dst);
}

// Negative filter
void negative(const cv::Mat &src, cv::Mat &dst) {
    dst.create(src.size(), src.type());
    dst = 255 - src;  // Inverts the color of the image
}

// bilateral filter
void applyBilateralFilter(const cv::Mat &src, cv::Mat &dst) {
    // Apply bilateral filter to smooth the image while keeping edges sharp
    cv::bilateralFilter(src, dst, 9, 75, 75);
}

// edge detection 
void applyEdgeDetection(const cv::Mat &src, cv::Mat &edges) {
    cv::Mat gray, sobelX, sobelY, grad;

    // Convert the source image to grayscale
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    // Apply Sobel filter in X and Y direction to detect edges
    sobelX3x3(gray, sobelX);
    sobelY3x3(gray, sobelY);

    // Compute the gradient magnitude from Sobel X and Sobel Y
    magnitude(sobelX, sobelY, grad);

    // Apply thresholding to get binary edges
    cv::threshold(grad, edges, 80, 255, cv::THRESH_BINARY_INV);
}

// apply color quantization
void applyColorQuantization(const cv::Mat &src, cv::Mat &dst, int levels = 8) {
    dst = src.clone();
    
    // Reduce the number of colors by quantizing each color channel
    int step = 256 / levels;
    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            for (int c = 0; c < 3; ++c) {
                dst.at<cv::Vec3b>(y, x)[c] = src.at<cv::Vec3b>(y, x)[c] / step * step + step / 2;
            }
        }
    }
}

// cartoon effect
void cartoonize(const cv::Mat &src, cv::Mat &dst) {
    // Initialize dst and ensure it's cleared
    dst.create(src.size(), src.type());
    dst = cv::Scalar(0, 0, 0);

    cv::Mat smoothed, edges, quantized;

    // Step 1: Apply bilateral filter to smooth the image but preserve edges
    applyBilateralFilter(src, smoothed);

    // Step 2: Detect edges using Sobel and thresholding
    applyEdgeDetection(src, edges);

    // Step 3: Apply color quantization for a cartoonish look
    applyColorQuantization(smoothed, quantized);

    // Ensure edges are converted to BGR before combining with quantized
    if (edges.channels() == 1) {
        cv::cvtColor(edges, edges, cv::COLOR_GRAY2BGR);
    }

    // Combine quantized image and edges using weighted addition
    cv::addWeighted(quantized, 0.8, edges, 0.2, 0, dst);
}



