/**
  * Copyright 2014, André R. Brodtkorb
  * Released under GPLv3
  */

#include <iostream>
#include <string>
#include <sstream>
#include <cstdlib>
#include <math.h>
#include <time.h>


#ifdef _WIN32
#include <sys/timeb.h>
#include <windows.h>
#else
#include <sys/time.h>
#include <unistd.h>
#endif


inline double getCurrentTime() {
#ifdef WIN32
    LARGE_INTEGER f;
    LARGE_INTEGER t;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return t.QuadPart/(double) f.QuadPart;
#else
    struct timeval tv;
    struct timezone tz;
    gettimeofday(&tv, &tz);
    return tv.tv_sec+tv.tv_usec*1e-6;
#endif
};

template<typename T>
T getFromCin() {
    std::string line;
    T retval;
    while (std::getline(std::cin, line)) {
        std::stringstream ss(line);
        if (ss >> retval && ss.eof()) {
            return retval;
        }
        std::cout << "You entered '" << line << "', which I was unable to interpret. Please try again" << std::endl;
    }
}

inline float generateRandomNumber() {
    return static_cast<float>(rand()) / static_cast <float> (RAND_MAX);
}

float computePi(int n_points) {
    int n_inside = 0;

    for (int i=0; i<n_points; ++i) {
        //Generate coordinate
        float x = generateRandomNumber();
        float y = generateRandomNumber();

        //Compute radius
        float r = sqrt(x*x + y*y);

        //Check if within circle
        if (r <= 1.0f) {
            ++n_inside;
        }
    }

    //Estimate Pi
    float pi = 4.0f * n_inside / static_cast<float>(n_points);

    return pi;
}

int main(int argc, char** argv) {
    std::cout << "Estimating value of Pi (Press CTRL+C to exit)" << std::endl;

    srand(time(NULL)); //Set random seed

    std::cout << "True value of Pi:   3.14159265359..." << std::endl;
    for(;;) {
        std::cout << "Please enter number of iterations: " << std::endl;
        int n_points = getFromCin<int>();
        double tic = getCurrentTime();
        float pi = computePi(n_points);
        double toc = getCurrentTime();
        std::cout << "Estimated Pi to be: " << std::fixed << pi << " in " << (toc-tic) << " seconds." << std::endl;
    }

    return 0;
}