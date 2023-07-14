//
// Created by ChuckySRB on 6/22/2023.
//

#include "../headers/Image_Processor_Optimised.h"
#include <emmintrin.h>
#include <immintrin.h>

ImageProcessorOptimised::ImageProcessorOptimised(const std::string& image) : ImageProcessor(image) {}


double ImageProcessorOptimised::grayscale() {

    auto start = std::chrono::high_resolution_clock::now();
    __m256 constV = _mm256_set1_ps(0.33); // Load the constant into a SIMD vector

    for (int y = 0; y < this->image.rows; ++y)
    {
        for (int x = 0; x < this->image.cols-8; x+= 8) // Process 8 pixels at a time using AVX
        {
            float R[8], G[8], B[8];
            for (int j = 0; j < 8; j++){
                Vec3b& pixels = this->image.at<cv::Vec3b>(y, x + j);
                B[j] = pixels[0];
                G[j] = pixels[1];
                R[j] = pixels[2];
            }


            __m256 redChannelFloat = _mm256_set_ps(R[7], R[6], R[5], R[4], R[3], R[2], R[1], R[0]);
            __m256 greenChannelFloat = _mm256_set_ps(G[7], G[6], G[5], G[4], G[3], G[2], G[1], G[0]);
            __m256 blueChannelFloat = _mm256_set_ps(B[7], B[6], B[5], B[4], B[4], B[2], B[1], B[0]);

            // Add the constant to the channel vectors
            blueChannelFloat = _mm256_mul_ps(blueChannelFloat, constV);
            greenChannelFloat = _mm256_mul_ps(greenChannelFloat, constV);
            redChannelFloat = _mm256_mul_ps(redChannelFloat, constV);

            // Clamp the channel vectors to the range [0, 255]
            __m256 greyChannelFloat = _mm256_add_ps(blueChannelFloat, greenChannelFloat);
            greyChannelFloat = _mm256_add_ps(greyChannelFloat, redChannelFloat);

            for (int j = 0; j < 8; j++) {
                Vec3b& pixels = this->image.at<cv::Vec3b>(y, x+j);
                float grey = greyChannelFloat.m256_f32[j];
                pixels[0] = grey;
                pixels[1] = grey;
                pixels[2] = grey;
            }
        }}
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double timeInSeconds = duration.count();
    this->iteration++;
    return timeInSeconds;
}

double ImageProcessorOptimised::add(double constant) {
    auto start = std::chrono::high_resolution_clock::now();
    __m256 constV = _mm256_set1_ps(constant); // Load the constant into a SIMD vector

    for (int y = 0; y < this->image.rows; ++y)
    {
        for (int x = 0; x < this->image.cols-8; x+= 8) // Process 8 pixels at a time using AVX
        {
            float R[8], G[8], B[8];
            for (int j = 0; j < 8; j++){
                Vec3b& pixels = this->image.at<cv::Vec3b>(y, x + j);
                B[j] = pixels[0];
                G[j] = pixels[1];
                R[j] = pixels[2];
            }


            __m256 redChannelFloat = _mm256_set_ps(R[7], R[6], R[5], R[4], R[3], R[2], R[1], R[0]);
            __m256 greenChannelFloat = _mm256_set_ps(G[7], G[6], G[5], G[4], G[3], G[2], G[1], G[0]);
            __m256 blueChannelFloat = _mm256_set_ps(B[7], B[6], B[5], B[4], B[4], B[2], B[1], B[0]);

            // Add the constant to the channel vectors
            blueChannelFloat = _mm256_add_ps(blueChannelFloat, constV);
            greenChannelFloat = _mm256_add_ps(greenChannelFloat, constV);
            redChannelFloat = _mm256_add_ps(redChannelFloat, constV);

            // Clamp the channel vectors to the range [0, 255]
            blueChannelFloat = _mm256_min_ps(blueChannelFloat, _mm256_set1_ps(255.0));
            greenChannelFloat = _mm256_min_ps(greenChannelFloat, _mm256_set1_ps(255.0));
            redChannelFloat = _mm256_min_ps(redChannelFloat, _mm256_set1_ps(255.0));

            for (int j = 0; j < 8; j++) {
                Vec3b& pixels = this->image.at<cv::Vec3b>(y, x+j);
                pixels[0] = blueChannelFloat.m256_f32[j];
                pixels[1] = greenChannelFloat.m256_f32[j];
                pixels[2] = redChannelFloat.m256_f32[j];
            }
        }}
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double timeInSeconds = duration.count();
    this->iteration++;
    return timeInSeconds;
}/*
double add2(double constant) {
    auto start = std::chrono::high_resolution_clock::now();
    __m256 constV = _mm256_set1_ps(constant); // Load the constant into a SIMD vector

    for (int y = 0; y < this->image.rows; ++y)
    {
        for (int x = 0; x < this->image.cols-8; x+= 8) // Process 8 pixels at a time using AVX
        {
            float R[8], G[8], B[8];
            for (int j = 0; j < 8; j++){
                Vec3b& pixels = this->image.at<cv::Vec3b>(y, x + j);
                B[j] = pixels[0];
                G[j] = pixels[1];
                R[j] = pixels[2];
            }

            uint8_t *image_pxl = image.data;

            __m256 pixels = _mm256_load_ps((__m256*)image_pxl + y*image.cols*4 + x*4);
            pixels = _mm256_add_ps(pixels, constV);
            pixels = _mm256_min_ps(pixels, _mm256_set1_ps(255.0));

            _mm256_store_((__m256i*)(d + i), vd);
        }}
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double timeInSeconds = duration.count();
    this->iteration++;
    return timeInSeconds;
}
*/
double ImageProcessorOptimised::sub(double constant) {
    auto start = std::chrono::high_resolution_clock::now();
    __m256 constV = _mm256_set1_ps(constant); // Load the constant into a SIMD vector

    for (int y = 0; y < this->image.rows; ++y)
    {
        for (int x = 0; x < this->image.cols-8; x+= 8) // Process 8 pixels at a time using AVX
        {
            float R[8], G[8], B[8];
            for (int j = 0; j < 8; j++){
                Vec3b& pixels = this->image.at<cv::Vec3b>(y, x + j);
                B[j] = pixels[0];
                G[j] = pixels[1];
                R[j] = pixels[2];
            }


            __m256 redChannelFloat = _mm256_set_ps(R[7], R[6], R[5], R[4], R[3], R[2], R[1], R[0]);
            __m256 greenChannelFloat = _mm256_set_ps(G[7], G[6], G[5], G[4], G[3], G[2], G[1], G[0]);
            __m256 blueChannelFloat = _mm256_set_ps(B[7], B[6], B[5], B[4], B[4], B[2], B[1], B[0]);

            // Add the constant to the channel vectors
            blueChannelFloat = _mm256_sub_ps(blueChannelFloat, constV);
            greenChannelFloat = _mm256_sub_ps(greenChannelFloat, constV);
            redChannelFloat = _mm256_sub_ps(redChannelFloat, constV);

            // Clamp the channel vectors to the range [0, 255]
            blueChannelFloat = _mm256_max_ps(blueChannelFloat, _mm256_set1_ps(0.0));
            greenChannelFloat = _mm256_max_ps(greenChannelFloat, _mm256_set1_ps(0.0));
            redChannelFloat = _mm256_max_ps(redChannelFloat, _mm256_set1_ps(0.0));

            for (int j = 0; j < 8; j++) {
                Vec3b& pixels = this->image.at<cv::Vec3b>(y, x+j);
                pixels[0] = blueChannelFloat.m256_f32[j];
                pixels[1] = greenChannelFloat.m256_f32[j];
                pixels[2] = redChannelFloat.m256_f32[j];
            }
        }}
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double timeInSeconds = duration.count();
    this->iteration++;
    return timeInSeconds;
}

double ImageProcessorOptimised::mul(double constant) {
    auto start = std::chrono::high_resolution_clock::now();
    __m256 constV = _mm256_set1_ps(constant); // Load the constant into a SIMD vector

    for (int y = 0; y < this->image.rows; ++y)
    {
        for (int x = 0; x < this->image.cols-8; x+= 8) // Process 8 pixels at a time using AVX
        {
            float R[8], G[8], B[8];
            for (int j = 0; j < 8; j++){
                Vec3b& pixels = this->image.at<cv::Vec3b>(y, x + j);
                B[j] = pixels[0];
                G[j] = pixels[1];
                R[j] = pixels[2];
            }


            __m256 redChannelFloat = _mm256_set_ps(R[7], R[6], R[5], R[4], R[3], R[2], R[1], R[0]);
            __m256 greenChannelFloat = _mm256_set_ps(G[7], G[6], G[5], G[4], G[3], G[2], G[1], G[0]);
            __m256 blueChannelFloat = _mm256_set_ps(B[7], B[6], B[5], B[4], B[4], B[2], B[1], B[0]);

            // Add the constant to the channel vectors
            blueChannelFloat = _mm256_mul_ps(blueChannelFloat, constV);
            greenChannelFloat = _mm256_mul_ps(greenChannelFloat, constV);
            redChannelFloat = _mm256_mul_ps(redChannelFloat, constV);

            // Clamp the channel vectors to the range [0, 255]
            blueChannelFloat = _mm256_min_ps(blueChannelFloat, _mm256_set1_ps(255.0));
            greenChannelFloat = _mm256_min_ps(greenChannelFloat, _mm256_set1_ps(255.0));
            redChannelFloat = _mm256_min_ps(redChannelFloat, _mm256_set1_ps(255.0));

            for (int j = 0; j < 8; j++) {
                Vec3b& pixels = this->image.at<cv::Vec3b>(y, x+j);
                pixels[0] = blueChannelFloat.m256_f32[j];
                pixels[1] = greenChannelFloat.m256_f32[j];
                pixels[2] = redChannelFloat.m256_f32[j];
            }
        }}
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double timeInSeconds = duration.count();
    this->iteration++;
    return timeInSeconds;
}

double ImageProcessorOptimised::div(double constant) {
    auto start = std::chrono::high_resolution_clock::now();
    __m256 constV = _mm256_set1_ps(constant); // Load the constant into a SIMD vector

    for (int y = 0; y < this->image.rows; ++y)
    {
        for (int x = 0; x < this->image.cols-8; x+= 8) // Process 8 pixels at a time using AVX
        {
            float R[8], G[8], B[8];
            for (int j = 0; j < 8; j++){
                Vec3b& pixels = this->image.at<cv::Vec3b>(y, x + j);
                B[j] = pixels[0];
                G[j] = pixels[1];
                R[j] = pixels[2];
            }


            __m256 redChannelFloat = _mm256_set_ps(R[7], R[6], R[5], R[4], R[3], R[2], R[1], R[0]);
            __m256 greenChannelFloat = _mm256_set_ps(G[7], G[6], G[5], G[4], G[3], G[2], G[1], G[0]);
            __m256 blueChannelFloat = _mm256_set_ps(B[7], B[6], B[5], B[4], B[4], B[2], B[1], B[0]);

            // Add the constant to the channel vectors
            blueChannelFloat = _mm256_div_ps(blueChannelFloat, constV);
            greenChannelFloat = _mm256_div_ps(greenChannelFloat, constV);
            redChannelFloat = _mm256_div_ps(redChannelFloat, constV);

            // Clamp the channel vectors to the range [0, 255]
            blueChannelFloat = _mm256_min_ps(blueChannelFloat, _mm256_set1_ps(255.0));
            greenChannelFloat = _mm256_min_ps(greenChannelFloat, _mm256_set1_ps(255.0));
            redChannelFloat = _mm256_min_ps(redChannelFloat, _mm256_set1_ps(255.0));

            for (int j = 0; j < 8; j++) {
                Vec3b& pixels = this->image.at<cv::Vec3b>(y, x+j);
                pixels[0] = blueChannelFloat.m256_f32[j];
                pixels[1] = greenChannelFloat.m256_f32[j];
                pixels[2] = redChannelFloat.m256_f32[j];
            }
        }}
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double timeInSeconds = duration.count();
    this->iteration++;
    return timeInSeconds;
}

double ImageProcessorOptimised::inv_sub(double constant) {
    auto start = std::chrono::high_resolution_clock::now();
    __m256 constV = _mm256_set1_ps(constant); // Load the constant into a SIMD vector

    for (int y = 0; y < this->image.rows; ++y)
    {
        for (int x = 0; x < this->image.cols-8; x+= 8) // Process 8 pixels at a time using AVX
        {
            float R[8], G[8], B[8];
            for (int j = 0; j < 8; j++){
                Vec3b& pixels = this->image.at<cv::Vec3b>(y, x + j);
                B[j] = pixels[0];
                G[j] = pixels[1];
                R[j] = pixels[2];
            }


            __m256 redChannelFloat = _mm256_set_ps(R[7], R[6], R[5], R[4], R[3], R[2], R[1], R[0]);
            __m256 greenChannelFloat = _mm256_set_ps(G[7], G[6], G[5], G[4], G[3], G[2], G[1], G[0]);
            __m256 blueChannelFloat = _mm256_set_ps(B[7], B[6], B[5], B[4], B[4], B[2], B[1], B[0]);

            // Add the constant to the channel vectors
            blueChannelFloat = _mm256_sub_ps(constV, blueChannelFloat);
            greenChannelFloat = _mm256_sub_ps(constV, greenChannelFloat);
            redChannelFloat = _mm256_sub_ps(constV, redChannelFloat);

            // Clamp the channel vectors to the range [0, 255]
            blueChannelFloat = _mm256_max_ps(blueChannelFloat, _mm256_set1_ps(0.0));
            greenChannelFloat = _mm256_max_ps(greenChannelFloat, _mm256_set1_ps(0.0));
            redChannelFloat = _mm256_max_ps(redChannelFloat, _mm256_set1_ps(0.0));

            for (int j = 0; j < 8; j++) {
                Vec3b& pixels = this->image.at<cv::Vec3b>(y, x+j);
                pixels[0] = blueChannelFloat.m256_f32[j];
                pixels[1] = greenChannelFloat.m256_f32[j];
                pixels[2] = redChannelFloat.m256_f32[j];
            }
        }}
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double timeInSeconds = duration.count();
    this->iteration++;
    return timeInSeconds;
}

double ImageProcessorOptimised::inv_div(double constant) {
    auto start = std::chrono::high_resolution_clock::now();
    __m256 constV = _mm256_set1_ps(constant); // Load the constant into a SIMD vector

    for (int y = 0; y < this->image.rows; ++y)
    {
        for (int x = 0; x < this->image.cols-8; x+= 8) // Process 8 pixels at a time using AVX
        {
            float R[8], G[8], B[8];
            for (int j = 0; j < 8; j++){
                Vec3b& pixels = this->image.at<cv::Vec3b>(y, x + j);
                B[j] = pixels[0];
                G[j] = pixels[1];
                R[j] = pixels[2];
            }


            __m256 redChannelFloat = _mm256_set_ps(R[7], R[6], R[5], R[4], R[3], R[2], R[1], R[0]);
            __m256 greenChannelFloat = _mm256_set_ps(G[7], G[6], G[5], G[4], G[3], G[2], G[1], G[0]);
            __m256 blueChannelFloat = _mm256_set_ps(B[7], B[6], B[5], B[4], B[4], B[2], B[1], B[0]);

            // Add the constant to the channel vectors
            blueChannelFloat = _mm256_div_ps(constV, blueChannelFloat);
            greenChannelFloat = _mm256_div_ps(constV, greenChannelFloat);
            redChannelFloat = _mm256_div_ps(constV, redChannelFloat);

            // Clamp the channel vectors to the range [0, 255]
            blueChannelFloat = _mm256_min_ps(blueChannelFloat, _mm256_set1_ps(255.0));
            greenChannelFloat = _mm256_min_ps(greenChannelFloat, _mm256_set1_ps(255.0));
            redChannelFloat = _mm256_min_ps(redChannelFloat, _mm256_set1_ps(255.0));

            for (int j = 0; j < 8; j++) {
                Vec3b& pixels = this->image.at<cv::Vec3b>(y, x+j);
                pixels[0] = blueChannelFloat.m256_f32[j];
                pixels[1] = greenChannelFloat.m256_f32[j];
                pixels[2] = redChannelFloat.m256_f32[j];
            }
        }}
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double timeInSeconds = duration.count();
    this->iteration++;
    return timeInSeconds;
}

double ImageProcessorOptimised::power(double constant) {
    auto start = std::chrono::high_resolution_clock::now();
    __m256 constV = _mm256_set1_ps(constant); // Load the constant into a SIMD vector

    for (int y = 0; y < this->image.rows; ++y)
    {
        for (int x = 0; x < this->image.cols-8; x+= 8) // Process 8 pixels at a time using AVX
        {
            float R[8], G[8], B[8];
            for (int j = 0; j < 8; j++){
                Vec3b& pixels = this->image.at<cv::Vec3b>(y, x + j);
                B[j] = pixels[0];
                G[j] = pixels[1];
                R[j] = pixels[2];
            }


            __m256 redChannelFloat = _mm256_set_ps(R[7], R[6], R[5], R[4], R[3], R[2], R[1], R[0]);
            __m256 greenChannelFloat = _mm256_set_ps(G[7], G[6], G[5], G[4], G[3], G[2], G[1], G[0]);
            __m256 blueChannelFloat = _mm256_set_ps(B[7], B[6], B[5], B[4], B[4], B[2], B[1], B[0]);

            // Add the constant to the channel vectors
            blueChannelFloat = _mm256_pow_ps(blueChannelFloat, constV);
            greenChannelFloat = _mm256_pow_ps(greenChannelFloat, constV);
            redChannelFloat = _mm256_pow_ps(redChannelFloat, constV);

            // Clamp the channel vectors to the range [0, 255]
            blueChannelFloat = _mm256_min_ps(blueChannelFloat, _mm256_set1_ps(255.0));
            greenChannelFloat = _mm256_min_ps(greenChannelFloat, _mm256_set1_ps(255.0));
            redChannelFloat = _mm256_min_ps(redChannelFloat, _mm256_set1_ps(255.0));

            for (int j = 0; j < 8; j++) {
                Vec3b& pixels = this->image.at<cv::Vec3b>(y, x+j);
                pixels[0] = blueChannelFloat.m256_f32[j];
                pixels[1] = greenChannelFloat.m256_f32[j];
                pixels[2] = redChannelFloat.m256_f32[j];
            }
        }}
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double timeInSeconds = duration.count();
    this->iteration++;
    return timeInSeconds;
}

double ImageProcessorOptimised::max(double constant) {
    auto start = std::chrono::high_resolution_clock::now();
    __m256 constV = _mm256_set1_ps(constant); // Load the constant into a SIMD vector

    for (int y = 0; y < this->image.rows; ++y)
    {
        for (int x = 0; x < this->image.cols-8; x+= 8) // Process 8 pixels at a time using AVX
        {
            float R[8], G[8], B[8];
            for (int j = 0; j < 8; j++){
                Vec3b& pixels = this->image.at<cv::Vec3b>(y, x + j);
                B[j] = pixels[0];
                G[j] = pixels[1];
                R[j] = pixels[2];
            }


            __m256 redChannelFloat = _mm256_set_ps(R[7], R[6], R[5], R[4], R[3], R[2], R[1], R[0]);
            __m256 greenChannelFloat = _mm256_set_ps(G[7], G[6], G[5], G[4], G[3], G[2], G[1], G[0]);
            __m256 blueChannelFloat = _mm256_set_ps(B[7], B[6], B[5], B[4], B[4], B[2], B[1], B[0]);

            // Add the constant to the channel vectors
            blueChannelFloat = _mm256_max_ps(blueChannelFloat, constV);
            greenChannelFloat = _mm256_max_ps(greenChannelFloat, constV);
            redChannelFloat = _mm256_max_ps(redChannelFloat, constV);


            for (int j = 0; j < 8; j++) {
                Vec3b& pixels = this->image.at<cv::Vec3b>(y, x+j);
                pixels[0] = blueChannelFloat.m256_f32[j];
                pixels[1] = greenChannelFloat.m256_f32[j];
                pixels[2] = redChannelFloat.m256_f32[j];
            }
        }}
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double timeInSeconds = duration.count();
    this->iteration++;
    return timeInSeconds;
}

double ImageProcessorOptimised::min(double constant) {
    auto start = std::chrono::high_resolution_clock::now();
    __m256 constV = _mm256_set1_ps(constant); // Load the constant into a SIMD vector

    for (int y = 0; y < this->image.rows; ++y)
    {
        for (int x = 0; x < this->image.cols-8; x+= 8) // Process 8 pixels at a time using AVX
        {
            float R[8], G[8], B[8];
            for (int j = 0; j < 8; j++){
                Vec3b& pixels = this->image.at<cv::Vec3b>(y, x + j);
                B[j] = pixels[0];
                G[j] = pixels[1];
                R[j] = pixels[2];
            }


            __m256 redChannelFloat = _mm256_set_ps(R[7], R[6], R[5], R[4], R[3], R[2], R[1], R[0]);
            __m256 greenChannelFloat = _mm256_set_ps(G[7], G[6], G[5], G[4], G[3], G[2], G[1], G[0]);
            __m256 blueChannelFloat = _mm256_set_ps(B[7], B[6], B[5], B[4], B[4], B[2], B[1], B[0]);

            // Add the constant to the channel vectors
            blueChannelFloat = _mm256_min_ps(blueChannelFloat, constV);
            greenChannelFloat = _mm256_min_ps(greenChannelFloat, constV);
            redChannelFloat = _mm256_min_ps(redChannelFloat, constV);


            for (int j = 0; j < 8; j++) {
                Vec3b& pixels = this->image.at<cv::Vec3b>(y, x+j);
                pixels[0] = blueChannelFloat.m256_f32[j];
                pixels[1] = greenChannelFloat.m256_f32[j];
                pixels[2] = redChannelFloat.m256_f32[j];
            }
        }}
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double timeInSeconds = duration.count();
    this->iteration++;
    return timeInSeconds;
}

double ImageProcessorOptimised::log() {
    auto start = std::chrono::high_resolution_clock::now();
    for (int y = 0; y < this->image.rows; ++y)
    {
        for (int x = 0; x < this->image.cols-8; x+= 8) // Process 8 pixels at a time using AVX
        {
            float R[8], G[8], B[8];
            for (int j = 0; j < 8; j++){
                Vec3b& pixels = this->image.at<cv::Vec3b>(y, x + j);
                B[j] = pixels[0];
                G[j] = pixels[1];
                R[j] = pixels[2];
            }


            __m256 redChannelFloat = _mm256_set_ps(R[7], R[6], R[5], R[4], R[3], R[2], R[1], R[0]);
            __m256 greenChannelFloat = _mm256_set_ps(G[7], G[6], G[5], G[4], G[3], G[2], G[1], G[0]);
            __m256 blueChannelFloat = _mm256_set_ps(B[7], B[6], B[5], B[4], B[4], B[2], B[1], B[0]);

            // Add the constant to the channel vectors
            blueChannelFloat = _mm256_log2_ps(blueChannelFloat);
            greenChannelFloat = _mm256_log2_ps(greenChannelFloat);
            redChannelFloat = _mm256_log2_ps(redChannelFloat);

            // Clamp the channel vectors to the range [0, 255]
            blueChannelFloat = _mm256_min_ps(blueChannelFloat, _mm256_set1_ps(255.0));
            greenChannelFloat = _mm256_min_ps(greenChannelFloat, _mm256_set1_ps(255.0));
            redChannelFloat = _mm256_min_ps(redChannelFloat, _mm256_set1_ps(255.0));

            for (int j = 0; j < 8; j++) {
                Vec3b& pixels = this->image.at<cv::Vec3b>(y, x+j);
                pixels[0] = blueChannelFloat.m256_f32[j];
                pixels[1] = greenChannelFloat.m256_f32[j];
                pixels[2] = redChannelFloat.m256_f32[j];
            }
        }}
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double timeInSeconds = duration.count();
    this->iteration++;
    return timeInSeconds;
}

double ImageProcessorOptimised::abs() {
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

double ImageProcessorOptimised::inversion() {
    auto start = std::chrono::high_resolution_clock::now();
    __m256 constV = _mm256_set1_ps(255); // Load the constant into a SIMD vector

    for (int y = 0; y < this->image.rows; ++y)
    {
        for (int x = 0; x < this->image.cols-8; x+= 8) // Process 8 pixels at a time using AVX
        {
            float R[8], G[8], B[8];
            for (int j = 0; j < 8; j++){
                Vec3b& pixels = this->image.at<cv::Vec3b>(y, x + j);
                B[j] = pixels[0];
                G[j] = pixels[1];
                R[j] = pixels[2];
            }


            __m256 redChannelFloat = _mm256_set_ps(R[7], R[6], R[5], R[4], R[3], R[2], R[1], R[0]);
            __m256 greenChannelFloat = _mm256_set_ps(G[7], G[6], G[5], G[4], G[3], G[2], G[1], G[0]);
            __m256 blueChannelFloat = _mm256_set_ps(B[7], B[6], B[5], B[4], B[4], B[2], B[1], B[0]);

            // Add the constant to the channel vectors
            blueChannelFloat = _mm256_sub_ps(constV, blueChannelFloat);
            greenChannelFloat = _mm256_sub_ps(constV, greenChannelFloat);
            redChannelFloat = _mm256_sub_ps(constV, redChannelFloat);

            for (int j = 0; j < 8; j++) {
                Vec3b& pixels = this->image.at<cv::Vec3b>(y, x+j);
                pixels[0] = blueChannelFloat.m256_f32[j];
                pixels[1] = greenChannelFloat.m256_f32[j];
                pixels[2] = redChannelFloat.m256_f32[j];
            }
        }}
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double timeInSeconds = duration.count();
    this->iteration++;
    return timeInSeconds;
}

double ImageProcessorOptimised::filter(int N, double **filter) {
    auto start = std::chrono::high_resolution_clock::now();
    int pad = N / 2;
    int chache_block = 128;
    unsigned long iter = 0;
    for (int y_block = 0 + pad; y_block < this->image.rows-pad; y_block += chache_block) {
        for (int x_block = 0 + pad; x_block < this->image.cols - pad; x_block += chache_block) {
            for (int y = y_block; (y < y_block+chache_block && y < this->image.rows-pad); ++y) {
                for (int x = x_block; (x <  x_block+chache_block && x < this->image.cols-pad); ++x)  {

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
    }}}
    std::cout << iter << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double timeInSeconds = duration.count();
    this->iteration++;
    return timeInSeconds;
}