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
#include <omp.h>


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

void setNumberOfThreads(int n_threads) {
    omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(n_threads); // Set number of threads
    #pragma omp parallel
    #pragma omp single
    {
        int actual_n_threads = omp_get_num_threads();
        if (actual_n_threads != n_threads) {
            std::cout << "Warning: OpenMP did not obay request, using " << actual_n_threads << " threads (did you remember to compile with OpenMP support?)." << std::endl;
        }
    }
}

float computePi(int n_points) {
    int n_inside = 0;

#pragma omp parallel for reduction(+:n_inside)
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

    #pragma omp parallel
    {
        srand(int(time(NULL)) ^ omp_get_thread_num()); //Set random seed, different for each OMP thread
    }
    
    //Set number of threads
    std::cout << "Please enter number of threads: ";
    int n_threads = getFromCin<int>();
    setNumberOfThreads(n_threads);
    
    std::cout << "True value of Pi:   3.14159265359..." << std::endl;
    for(;;) {
        std::cout << "Please enter number of iterations: ";
        int n_points = getFromCin<int>();

        double tic = getCurrentTime();

        float pi=0;
        {
            pi = computePi(n_points);
        }
        double toc = getCurrentTime();
        std::cout << "Estimated Pi to be: " << std::fixed << pi << " in " << (toc-tic) << " seconds." << std::endl;
    }
    
    return 0;
}