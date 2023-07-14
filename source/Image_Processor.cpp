//
// Created by ChuckySRB on 6/22/2023.
//

#include "../headers/Image_Processor.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>


using namespace cv;
using namespace std;

ImageProcessor::ImageProcessor(const std::string &image_path) {
    this->image = imread(image_path, cv::IMREAD_COLOR);
    this->iteration = 0;
    if (this->image.empty()) {
        std::cerr << "Failed to load the image: " << image_path << std::endl;
    }
}


void ImageProcessor::displayInfo() {

}

void ImageProcessor::print_options() {
    cout << "Dostupne opcije funkcija:" << endl;
    for (int i = 1; i <= N_Options; i++){
        cout << i << ") " << this->options[i-1] << endl;
    }
}

void ImageProcessor::show_img() {
    if(created()) {
        namedWindow("Edited Image" + to_string(this->iteration), cv::WINDOW_NORMAL); // Create a window
        imshow("Edited Image" + to_string(this->iteration), this->image); // Show the image in the window

        // Wait for a key press to exit
        cv::waitKey(0);
    }
}

void ImageProcessor::save_img(string img_path){
    if(created()) {
        cv::imwrite(img_path, this->image);
        std::cout << "Slika sacuvana na: " << img_path << std::endl;
    }
}

bool ImageProcessor::created() {
    return !this->image.empty();
}

double ImageProcessor::method_caller(string method) {

    /* SABIRANJE", "ODUZIMANJE", "INVERZNO_ODUZIMANJE", "MNOZENJE",
     * "DELJENJE", "INVERZNO_DELJENJE", "POWER", "LOG", "ABS", "MAX",
     * "MIN", "INVERZIJA", "SIVA", "FILTER"*/

    if (method == "FILTER") {
        int N;
        cout << "Unesite velicinu ivice kvadrata filtera:" << endl;
        cin >> N;
        cout << "Unesite elemente matrice" << endl;

        if (N >= 0) {
            auto **filter = new double*[N];
            for (int i = 0; i < N; i++) {
                filter[i] = new double[N];
                for (int j = 0; j < N; j++) {
                    cin >> filter[i][j];
                }
            }
            return this->filter(N,filter);
        }
        else{
            cout << "Neispravna veličina matrice" << endl;
        }
    }
    else if (method == "LOG"){
        return this->log();
    }
    else if (method == "ABS"){
        return this->abs();
    }
    else if (method == "INVERZIJA"){
        return this->inversion();
    }
    else if (method == "SIVA"){
        return this->grayscale();
    }
    else {
        double konstanta;
        cout << "Unesite konstantu:" << endl;
        cin >> konstanta;
        if (method == "SABIRANJE") {
            return this->add(konstanta);
        } else if (method == "ODUZIMANJE") {
            return this->sub(konstanta);
        } else if (method == "INVERZNO_ODUZIMANJE") {
            return this->inv_sub(konstanta);
        } else if (method == "MNOZENJE") {
            return this->mul(konstanta);
        } else if (method == "DELJENJE") {
            return this->div(konstanta);
        } else if (method == "INVERZNO_DELJENJE") {
            return this->inv_div(konstanta);
        } else if (method == "POWER") {
            return this->power(konstanta);
        } else if (method == "MAX") {
            return this->max(konstanta);
        } else if (method == "MIN") {
            return this->min(konstanta);
        } else {
            std::cout << "Nepostojeca metoda!" << std::endl;
        }
    }
    return 0;
}

