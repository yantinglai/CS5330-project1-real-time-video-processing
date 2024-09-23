/*
 * Yanting Lai
 * Date: 2024-09-22
 * CS 5330 - Computer Vision
 * 
 * Implementation of ten different image processing filters:
 * 1. Grayscale (both default and custom)
 * 2. Sepia tone with vignetting
 * 3. Gaussian Blur (two versions: naive and optimized)
 * 4. Sobel X and Sobel Y edge detection
 * 5. Gradient Magnitude
 * 6. Blur and Quantize
 * 7. Embossing effect
 * 8. Negative image transformation
 * 9. Median Filter
 * 10. Cartoonization (Extension)
 */

#ifndef FILTERS_HPP
#define FILTERS_HPP

#include <opencv2/opencv.hpp>


/**
 * @brief Create a grayscale image using the red channel 
 * in reverse (255 - red channel) as the grayscale tone.
 * This function converts a color image to a 
 * custom grayscale image by modifying all channels 
 * based on the red channel.
 * 
 * @param src The input color image.
 * @param dst The output grayscale image.
 * @return int Returns 0 on success, -1 if the input image is empty.
 */
int greyscale(cv::Mat &src, cv::Mat &dst);


/**
 * @brief Applies a sepia tone filter with vignetting effect.
 * The function applies a sepia tone transformation to the color image 
 * and gradually darkens the image towards the edges, 
 * simulating a vignette effect.
 * 
 * @param src The input color image.
 * @param dst The output sepia-toned image with vignetting effect.
 * @return int Returns 0 on success, -1 if the input image is empty.
 */
int sepia(cv::Mat &src, cv::Mat &dst);

/**
 * @brief Apply a naive 5x5 Gaussian blur filter.
 * This function applies a 5x5 Gaussian blur 
 * to each pixel in the image by averaging surrounding pixels with different weights.
 * 
 * @param src The input color image.
 * @param dst The output blurred image.
 * @return int Returns 0 on success, -1 if the input image is empty.
 */
int blur5x5_1(cv::Mat &src, cv::Mat &dst);

/**
 * @brief Apply a separable 5x5 Gaussian blur filter for optimized performance.
 * The function applies two 1x5 Gaussian filters 
 * (horizontal and vertical) in sequence, 
 * reducing computational complexity.
 */
int blur5x5_2(cv::Mat &src, cv::Mat &dst);

/**
 * @brief Apply the Sobel X filter for edge detection in the horizontal direction.
 * This function applies a 3x3 Sobel filter 
 * in the X direction to detect vertical edges in the image.
 * 
 */
int sobelX3x3(const cv::Mat &src, cv::Mat &dst);

/**
 * @brief Apply the Sobel Y filter for edge detection in the vertical direction.
 * This function applies a 3x3 Sobel filter in the Y direction 
 * to detect horizontal edges in the image.
 */
int sobelY3x3(const cv::Mat &src, cv::Mat &dst);

/**
 * @brief Compute the gradient magnitude of the image using the Sobel X and Sobel Y filters.
 * The gradient magnitude is calculated by combining the results of Sobel X and Sobel Y filters to create a single edge-detected image.
 * 
 * @param sx The Sobel X edge-detected image.
 * @param sy The Sobel Y edge-detected image.
 * @param dst The output gradient magnitude image.
 */
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);

/**
 * @brief Blur and quantize the image to reduce the number of colors, creating a cartoonish effect.
 * The image is blurred using a Gaussian filter, 
 * and then each color channel is quantized into a fixed number of levels.
 * 
 * @param src The input color image.
 * @param dst The output blurred and quantized image.
 * @param levels The number of quantization levels to apply to the image.
 */
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels);

/**
 * @brief Apply an emboss filter to the image using Sobel X and Y filters.
 * This function creates an embossed effect by 
 * combining Sobel X and Y results and adding a bias 
 * to shift the result into a valid range.
 * 
 * @param src The input color image.
 * @param dst The output embossed image.
 */
void emboss(const cv::Mat &src, cv::Mat &dst);

/**
 * @brief Apply a negative filter to invert the colors of the image.
 * This function inverts all color channels 
 * by subtracting the pixel values from 255, 
 * creating a negative image.
 * 
 * @param src The input color image.
 * @param dst The output negative image.
 */
void negative(const cv::Mat &src, cv::Mat &dst);

/**
 * @brief Apply a bilateral filter to smooth the image while preserving edges.
 * This function applies a bilateral filter to 
 * reduce noise and smooth the image, but keeps sharp edges intact.
 */
void applyBilateralFilter(const cv::Mat &src, cv::Mat &dst);

/**
 * @brief Apply edge detection using Sobel filters and thresholding to create binary edges.
 * The function applies Sobel filters in the X and Y directions, 
 * computes the gradient magnitude, and applies thresholding to obtain strong binary edges.
 * 
 * @param src The input color image.
 * @param edges The output binary edge-detected image.
 */
void applyEdgeDetection(const cv::Mat &src, cv::Mat &edges);

/**
 * @brief Apply color quantization to reduce the number of colors 
 * in the image for a cartoon-like effect.
 * This function reduces the number of color levels 
 * in each channel of the image, 
 * creating a more simplified, cartoonish look.
 * 
 * @param src The input color image.
 * @param dst The output color-quantized image.
 * @param levels The number of quantization levels to apply to the image.
 */
void applyColorQuantization(const cv::Mat &src, cv::Mat &dst, int levels);

/**
 * @brief Cartoonize the image by combining smoothing, edge detection, and color quantization.
 * The function applies a bilateral filter for smoothing, 
 * edge detection for prominent edges, 
 * and color quantization to reduce the color palette, 
 * creating a cartoon-like effect.
 * 
 * @param src The input color image.
 * @param dst The output cartoonized image.
 */
void cartoonize(const cv::Mat &src, cv::Mat &dst);


#endif // FILTERS_HPP