/*
 * Name: Yanting Lai
 * Date: September 17, 2024
 * 
 * * Purpose: 
 * This file reads an image from a given path, displays it in a window, 
 * and allows the user to toggle between various filters using key presses.
 * The user can press:
 * - 's' to toggle a sepia filter on/off
 * - 'b' to apply a 5x5 naive blur filter
 * - 'n' to apply a 5x5 optimized blur filter
 * The user can press 'q' to exit the program.
 * 
 */

#include <opencv2/opencv.hpp>
#include "filters.hpp"
#include <iostream> 


int main() {
    // Read the image file from the provided path
    cv::Mat img = cv::imread("/Users/sundri/Desktop/CS5330/Project/Project1/meow.jpg");
    cv::Mat sepiaImg;  // To store the sepia-filtered image
    cv::Mat blurImg1;  // To store the blur1-filtered image
    cv::Mat blurImg2;  // To store the blur2-filtered image
    bool sepiaApplied = false;  // Track if the sepia filter is applied
    bool blur1Applied = false;  // Track if the first blur filter is applied
    bool blur2Applied = false;  // Track if the second blur filter is applied

    // Check if the image was successfully loaded
    if (img.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return -1;
    }

    // Display the original image in a window named "Image Display"
    cv::imshow("Image Display", img);

    while (true) {
        char key = cv::waitKey(0);  // Wait indefinitely for key press

        // Exit the loop if 'q' is pressed
        if (key == 'q') {
            break;  
        } else if (key == 's') {
            // Toggle sepia filter
            if (sepiaApplied) {
                cv::imshow("Image Display", img);  // Show original image
                sepiaApplied = false;  // Update the flag
            } else {
                sepia(img, sepiaImg);  // Apply sepia filter
                cv::imshow("Image Display", sepiaImg);  // Display the sepia image
                sepiaApplied = true;  // Update the flag
            }
        } else if (key == 'b') {
            // Apply the naive blur filter
            double start = cv::getTickCount();  // Start timing
            blur5x5_1(img, blurImg1);  // Apply the naive blur function
            double end = cv::getTickCount();  // End timing
            std::cout << "blur5x5_1 Time: " << (end - start) / cv::getTickFrequency() << " seconds" << std::endl;
            cv::imshow("Image Display", blurImg1);  // Display the blurred image
            blur1Applied = true;  // Update the flag
        } else if (key == 'n') {
            // Apply the optimized blur filter
            double start = cv::getTickCount();  // Start timing
            blur5x5_2(img, blurImg2);  // Apply the optimized blur function
            double end = cv::getTickCount();  // End timing
            std::cout << "blur5x5_2 Time: " << (end - start) / cv::getTickFrequency() << " seconds" << std::endl;

            cv::imshow("Image Display", blurImg2);  // Display the blurred image
            blur2Applied = true;  // Update the flag
        }
    }

    return 0;
}
