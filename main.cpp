#include "headers/Image_Processor.h"
#include "headers/Image_Processor_Optimised.h"
#include "headers/Image_Processor_Not_Optimised.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>


using namespace cv;
using namespace std;
int main()
{


    string image_src;
    cout << "Unesite putanju do slike:\n";
    cin >> image_src;

    ImageProcessor* OP = new ImageProcessorOptimised(image_src);
    if (!OP->created()){
        cout << "Losa putanja do slike!" << endl;
        return 0;
    }
    ImageProcessor* NOOP = new ImageProcessorNotOptimised(image_src);
    OP->print_options();
    while (true){
        string method;
        cout << "Unesite funkciju koju zelite da izvrsite nad slikom:\n";
        cin >> method;
        if (method == "NONE" || method == "EXIT")
            break;
        double time_noop = NOOP->method_caller(method);
        double time_op = OP->method_caller(method);

        cout << "Time needed for Not Optimised " << time_noop << endl;
        cout << "Time needed for Optimised: " << time_op << endl;
        cout << "Upgraded speed " << time_noop / time_op << " times"<<endl;
        NOOP->save_img("../edited_img.jpg");

    }
    // ../images/karikatura.png
    // ../images/joker.jpg
    /*

     1 0 1 0 1 1 0 1 0
     0 -1 0 -1 0 0 -1 0 -1
     1 -2 3 -2 1 1 -2 3 -2
     0 -1 0 -1 0 0 -1 0 -1
     1 0 1 0 1 1 0 1 0
     0 -1 0 -1 0 0 -1 0 -1
     1 -2 3 -2 1 1 -2 3 -2
     0 -1 0 -1 0 0 -1 0 -1
     1 0 1 0 1 1 0 1 0

     1 0 1 0 1
     0 -1 0 -1 0
     1 -2 3 -2 0
     0 -1 0 -1 0
     1 0 1 0 1

     1 0 1
     -2 3 -2
     1 0 1

     */


    return 0;
}