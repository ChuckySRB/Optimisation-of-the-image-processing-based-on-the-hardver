//
// Created by ChuckySRB on 6/22/2023.
//
#include "../headers/Image_Processor_Not_Optimised.h"
#include <chrono>
#include <cmath>

ImageProcessorNotOptimised::ImageProcessorNotOptimised(const std::string& image) : ImageProcessor(image) {}


double ImageProcessorNotOptimised::grayscale() {
    auto start = std::chrono::high_resolution_clock::now();
    for (int y = 0; y < this->image.rows; ++y) {
        for (int x = 0; x < this->image.cols; ++x) {
            Vec3b& pixel = this->image.at<cv::Vec3b>(y, x);

            unsigned char grayValue = (pixel[0] + pixel[1] + pixel[2]) / 3;
            pixel[0] = grayValue;   // Blue channel
            pixel[1] = grayValue;   // Green channel
            pixel[2] = grayValue;   // Red channel
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double timeInSeconds = duration.count();
    this->iteration++;
    return timeInSeconds;
}

double ImageProcessorNotOptimised::add(double constant) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int y = 0; y < this->image.rows; ++y) {
        for (int x = 0; x < this->image.cols; ++x) {
            Vec3b& pixel = this->image.at<cv::Vec3b>(y, x);

            pixel[0] = (double(pixel[0]) + constant > 255 ? 255 : double(pixel[0]) + constant);   // Blue channel
            pixel[1] = (double(pixel[1]) + constant > 255 ? 255 : double(pixel[1]) + constant);  // Green channel
            pixel[2] = (double(pixel[2]) + constant > 255 ? 255 : double(pixel[2]) + constant);   // Red channel
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double timeInSeconds = duration.count();
    this->iteration++;
    return timeInSeconds;
}

double ImageProcessorNotOptimised::sub(double constant) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int y = 0; y < this->image.rows; ++y) {
        for (int x = 0; x < this->image.cols; ++x) {
            Vec3b& pixel = this->image.at<cv::Vec3b>(y, x);

            pixel[0] = (double(pixel[0]) - constant < 0 ? 0 : double(pixel[0]) - constant);   // Blue channel
            pixel[1] = (double(pixel[1]) - constant < 0 ? 0 : double(pixel[1]) - constant);  // Green channel
            pixel[2] = (double(pixel[2]) - constant < 0 ? 0 : double(pixel[2]) - constant);   // Red channel
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double timeInSeconds = duration.count();
    this->iteration++;
    return timeInSeconds;
}

double ImageProcessorNotOptimised::mul(double constant) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int y = 0; y < this->image.rows; ++y) {
        for (int x = 0; x < this->image.cols; ++x) {
            Vec3b& pixel = this->image.at<cv::Vec3b>(y, x);
            pixel[0] = (double(pixel[0]) * constant > 255 ? 255 : double(pixel[0]) * constant);   // Blue channel
            pixel[1] = (double(pixel[1]) * constant > 255 ? 255 : double(pixel[1]) * constant);  // Green channel
            pixel[2] = (double(pixel[2]) * constant > 255 ? 255 : double(pixel[2]) * constant);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double timeInSeconds = duration.count();
    this->iteration++;
    return timeInSeconds;
}

double ImageProcessorNotOptimised::div(double constant) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int y = 0; y < this->image.rows; ++y) {
        for (int x = 0; x < this->image.cols; ++x) {
            Vec3b& pixel = this->image.at<cv::Vec3b>(y, x);
            pixel[0] = (double(pixel[0]) / constant > 255 ? 255 : double(pixel[0]) / constant);   // Blue channel
            pixel[1] = (double(pixel[1]) / constant > 255 ? 255 : double(pixel[1]) / constant);  // Green channel
            pixel[2] = (double(pixel[2]) / constant > 255 ? 255 : double(pixel[2]) / constant);   // Red channel
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double timeInSeconds = duration.count();
    this->iteration++;
    return timeInSeconds;
}

double ImageProcessorNotOptimised::inv_sub(double constant) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int y = 0; y < this->image.rows; ++y) {
        for (int x = 0; x < this->image.cols; ++x) {
            Vec3b& pixel = this->image.at<cv::Vec3b>(y, x);
            pixel[0] = (constant - double(pixel[0]) < 0 ? 0 : constant - double(pixel[0]));   // Blue channel
            pixel[1] = (constant - double(pixel[1]) < 0 ? 0 : constant - double(pixel[1]));   // Green channel
            pixel[2] = (constant - double(pixel[2]) < 0 ? 0 : constant - double(pixel[2]));   // Red channel
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double timeInSeconds = duration.count();
    this->iteration++;
    return timeInSeconds;
}

double ImageProcessorNotOptimised::inv_div(double constant) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int y = 0; y < this->image.rows; ++y) {
        for (int x = 0; x < this->image.cols; ++x) {
            Vec3b& pixel = this->image.at<cv::Vec3b>(y, x);
            pixel[0] = (constant / double(pixel[0]) > 255 ? 255 : constant / double(pixel[0]));    // Blue channel
            pixel[1] = (constant / double(pixel[1]) > 255 ? 255 : constant / double(pixel[1]));   // Green channel
            pixel[2] = (constant / double(pixel[2]) > 255 ? 255 : constant / double(pixel[2]));   // Red channel
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double timeInSeconds = duration.count();
    this->iteration++;
    return timeInSeconds;
}

double ImageProcessorNotOptimised::power(double constant) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int y = 0; y < this->image.rows; ++y) {
        for (int x = 0; x < this->image.cols; ++x) {
            Vec3b& pixel = this->image.at<cv::Vec3b>(y, x);
            pixel[0] = (pow(pixel[0], constant) > 255 ? 255 : pow(pixel[0], constant));   // Blue channel
            pixel[1] = (pow(pixel[1], constant) > 255 ? 255 : pow(pixel[1], constant));   // Green channel
            pixel[2] = (pow(pixel[2], constant) > 255 ? 255 : pow(pixel[2], constant));   // Red channel
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double timeInSeconds = duration.count();
    this->iteration++;
    return timeInSeconds;
}

double ImageProcessorNotOptimised::max(double constant) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int y = 0; y < this->image.rows; ++y) {
        for (int x = 0; x < this->image.cols; ++x) {
            Vec3b& pixel = this->image.at<cv::Vec3b>(y, x);
            pixel[0] = (pixel[0] < constant ? pixel[0] : constant);   // Blue channel
            pixel[1] = (pixel[1] < constant ? pixel[1] : constant);   // Green channel
            pixel[2] = (pixel[2] < constant ? pixel[2] : constant);   // Red channel
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double timeInSeconds = duration.count();
    this->iteration++;
    return timeInSeconds;
}

double ImageProcessorNotOptimised::min(double constant) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int y = 0; y < this->image.rows; ++y) {
        for (int x = 0; x < this->image.cols; ++x) {
            Vec3b& pixel = this->image.at<cv::Vec3b>(y, x);
            pixel[0] = (pixel[0] > constant ? pixel[0] : constant);   // Blue channel
            pixel[1] = (pixel[1] > constant ? pixel[1] : constant);   // Green channel
            pixel[2] = (pixel[2] > constant ? pixel[2] : constant);   // Red channel
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double timeInSeconds = duration.count();
    this->iteration++;
    return timeInSeconds;
}

double ImageProcessorNotOptimised::log() {
    auto start = std::chrono::high_resolution_clock::now();
    for (int y = 0; y < this->image.rows; ++y) {
        for (int x = 0; x < this->image.cols; ++x) {
            Vec3b& pixel = this->image.at<cv::Vec3b>(y, x);
            pixel[0] = (std::log(pixel[0]) > 255 ? 255 : std::log(pixel[0]));   // Blue channel
            pixel[1] = (std::log(pixel[1]) > 255 ? 255 : std::log(pixel[1]));   // Green channel
            pixel[2] = (std::log(pixel[2]) > 255 ? 255 : std::log(pixel[2]));   // Red channel
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double timeInSeconds = duration.count();
    this->iteration++;
    return timeInSeconds;
}

double ImageProcessorNotOptimised::abs() {
    auto start = std::chrono::high_resolution_clock::now();
    for (int y = 0; y < this->image.rows; ++y) {
        for (int x = 0; x < this->image.cols; ++x) {
            Vec3b& pixel = this->image.at<cv::Vec3b>(y, x);
            pixel[0] = std::abs(pixel[0]);   // Blue channel
            pixel[1] = std::abs(pixel[1]);   // Green channel
            pixel[2] = std::abs(pixel[2]);   // Red channel
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double timeInSeconds = duration.count();
    this->iteration++;
    return timeInSeconds;
}

double ImageProcessorNotOptimised::inversion() {
    auto start = std::chrono::high_resolution_clock::now();
    for (int y = 0; y < this->image.rows; ++y) {
        for (int x = 0; x < this->image.cols; ++x) {
            Vec3b& pixel = this->image.at<cv::Vec3b>(y, x);

            unsigned char max_val = 255;
            pixel[0] = max_val - pixel[0];   // Blue channel
            pixel[1] = max_val - pixel[1];   // Green channel
            pixel[2] = max_val - pixel[2];   // Red channel
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double timeInSeconds = duration.count();
    this->iteration++;
    return timeInSeconds;
}

double ImageProcessorNotOptimised::filter(int N, double **filter) {
    auto start = std::chrono::high_resolution_clock::now();
    int pad = N / 2;
    unsigned long iter = 0;
    for (int y = 0 + pad; y < this->image.rows-pad; ++y) {
        for (int x = 0 + pad; x < this->image.cols - pad; ++x) {
            double blue = 0;
            double green = 0;
            double red = 0;
            for (int filter_y = y - pad, i = 0; filter_y <= y+pad; ++filter_y, i++) {
                for (int filter_x = x - pad, j= 0; filter_x <= x+pad; ++filter_x, j++) {
                    Vec3b& pixel = this->image.at<cv::Vec3b>(filter_y, filter_x);
                    blue += (double(pixel[0])*filter[i][j]);
                    green += (double(pixel[1])*filter[i][j]);
                    red += (double(pixel[2])*filter[i][j]);
                    iter++;
                 }
            }
            Vec3b& pixel = this->image.at<cv::Vec3b>(y, x);
            pixel[0] = (blue > 255 ? 255 : (blue < 0 ? 0 : blue)); // Blue channel
            pixel[1] = (green > 255 ? 255 : (green < 0 ? 0 : green)); // Green channel
            pixel[2] = (red > 255 ? 255 : (red < 0 ? 0 : red)); // Red channel
        }
    }
    std::cout << iter << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double timeInSeconds = duration.count();
    this->iteration++;
    return timeInSeconds;
}


