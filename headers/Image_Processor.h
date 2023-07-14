//
// Created by ChuckySRB on 6/22/2023.
//
#ifndef PROJEKAT_IMAGE_PROCESSOR_H
#define PROJEKAT_IMAGE_PROCESSOR_H

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>

using namespace cv;


class ImageProcessor {
protected:
    Mat image;
    int iteration;
    static const int N_Options = 14;
    std::string options[N_Options] = {"SABIRANJE", "ODUZIMANJE", "INVERZNO_ODUZIMANJE", "MNOZENJE",
                               "DELJENJE", "INVERZNO_DELJENJE", "POWER", "LOG", "ABS", "MAX",
                               "MIN", "INVERZIJA", "SIVA", "FILTER"};
public:
    explicit ImageProcessor(const std::string& image);
    virtual ~ImageProcessor() = default;

    void displayInfo();
    void save_img(std::string img_path);
    void show_img();
    void print_options();
    bool created();
    double method_caller(std::string method);
    virtual double add(double constant) = 0;
    virtual double sub(double constant) = 0;
    virtual double mul(double constant) = 0;
    virtual double div(double constant) = 0;
    virtual double inv_sub(double constant) = 0;
    virtual double inv_div(double constant) = 0;
    virtual double power(double constant) = 0;
    virtual double max(double constant) = 0;
    virtual double min(double constant) = 0;
    virtual double log() = 0;
    virtual double abs() = 0;
    virtual double inversion() = 0;
    virtual double grayscale() = 0;
    virtual double filter(int N, double** filter) = 0;

};


#endif //PROJEKAT_IMAGE_PROCESSOR_H
