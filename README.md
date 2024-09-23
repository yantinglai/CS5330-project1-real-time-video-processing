# Name

**Yanting Lai**

# Group Members

Individual project submission

---

## Project Information:

**Project**: CS 5330 - Computer Vision, Project 1, Real Time Video Processing

---

## Files Submitted:

- `imgDisplay.cpp`
- `vidDisplay.cpp`
- `filters.cpp`
- `filters.h`
- `README.txt`
- `CS5330-P1-Real time video processing.pdf`

---

## Video and Image Files Submission:

- Please download the video and image files using the following Google Drive links:

- **Video (CPA_Intro_Yanting.mp4)**: [Download here](https://drive.google.com/file/d/1y3vYQvE1gBV2KlFWVFiDC_qT3hLDInTx/view?usp=sharing)
- **Image (meow.jpg)**: [Download here](https://drive.google.com/file/d/1ahtKobRvRnlbrhNxGENsmiWs8kGSQ_Y1/view?usp=sharing)

- Save these files locally and modify the code to use the correct file paths for the video and image in your environment.

## Example:

```cpp
cv::VideoCapture capdev("path_to_video/CPA_Intro_Yanting.mp4");
cv::Mat img = cv::imread("path_to_image/meow.jpg");
```

---

## Development Environment:

- **Operating System**: macOS Ventura (13.x)
- **IDE**: Visual Studio Code

---

# Build Instructions

## Step 1: Install Required Dependencies

Before building the project, ensure you have **CMake** and **OpenCV** installed on your machine.

- Install **CMake** by following the instructions from: https://cmake.org/.
- For **macOS**, install **OpenCV** using the following command (via Homebrew):

```bash
brew install opencv
```

## Step 2: Create a Build Directory

```
mkdir build
cd build
```

## Step 3: Run CMake to Generate Build Files

```
cmake ..
```

## Step 4: Build the Executables

```
make
```

## Step 5: Run the Program

- To run `imgDisplay`

```
./imgDisplay imgDisplay
```

- To run `vidDisplay`

```
./vidDisplay vidDisplay
```

---

## Time Travel Days:

I am **not using** travel days for this project.

---

## Submission Notes:

The project contains the implementation of various image processing filters such as grayscale, sepia, blur, Sobel edge detection, and an extension of cartoonization. All filters can be toggled using keypresses in the live video application.

The code was tested on macOS using Visual Studio Code and successfully compiled using the provided Makefile.
