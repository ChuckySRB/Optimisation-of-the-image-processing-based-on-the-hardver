//
// Created by ChuckySRB on 6/22/2023.
//

#ifndef PROJEKAT_IMAGE_PROCESSOR_NOT_OPTIMISED_H
#define PROJEKAT_IMAGE_PROCESSOR_NOT_OPTIMISED_H

#include "Image_Processor.h"

// ImageProcessorNotOptimised class
class ImageProcessorNotOptimised : public ImageProcessor {
public:
    explicit ImageProcessorNotOptimised(const std::string& image);

    double add(double constant) override;
    double sub(double constant) override;
    double mul(double constant) override;
    double div(double constant) override;
    double inv_sub(double constant) override;
    double inv_div(double constant) override;
    double power(double constant) override;
    double max(double constant) override;
    double min(double constant) override;
    double log() override;
    double abs() override;
    double inversion() override;
    double grayscale() override;
    double filter(int N, double** filter) override;
};

#endif //PROJEKAT_IMAGE_PROCESSOR_NOT_OPTIMISED_H
