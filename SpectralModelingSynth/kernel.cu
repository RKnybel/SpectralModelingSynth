
#define NOMINMAX
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string.h>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <queue>
#include <Windows.h>
#include <mmsystem.h>
#include <chrono>
#include <fstream>

const int sampleRate = 44100;
const int d_nInstrument = 100; // can also think of instruments as paraphonic oscillators
const int d_nPartials = 50;

//partial frequency array. Each of the 50 sine wave partials gets a frequency
float h_freqs[50] = { 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000 };
float* d_freqs;

//partial amplitude array. Each of the 50 sine wave partials gets an amplitude
float h_amps[50] = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
float* d_amps;

//GRANULATITIES
// 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0
int granularities[] = {882, 4410, 8820, 44100, 88200, 441000, 882000};
float granularity_times[] = { 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0 };

int numSamples = 882;
//float* h_samples = (float*)malloc(sizeof(float) * numSamples);
float* d_samples;

//partial table. I want to learn what this does.
int table_nInstruments = 50;
int table_nPartials = 50;
float* d_partialEnergy;
float* h_partialArray = (float*)malloc(sizeof(float)* table_nInstruments* table_nPartials);
float* d_partialArray;
cudaTextureObject_t texPartials;

//sount fonts? samples? Arrays of normalized partial data to use for textures:
// 
float partialEnergy[50] = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };

//// new algorithm (3?) - 
//__global__ void new_method(
//    float* d_freqs, // pointer to frequency array in global memory
//    float* d_amps, // pointer to amplitude array in global memory
//    char* d_samples, // pointer to sample array in global memory
//    int streamNo, // the stream index
//    float timeOffset // the time offset of current granularity
//) {
//    const float Pi = 3.141592;
//    const float UnitTime = 0.01;
//    float t = (float)(threadIdx.x) / sampleRate // sampleRate : 44100Hz
//        + (float)streamNo * UnitTime + timeOffset; // UnitTime : 0.01s
//    float sum = 0;
//    float partialAmp = 1.0 / d_nPartials;
//    // blockIdx.x: 0 for left channel, 1 for right channel.
//    if (blockIdx.x == 0) {
//        // block 0 computes the instrument index 0 – 49 for left channel
//        // d_nInstrument : number of instruments. (N=100)
//        for (int k = 0; k < d_nPartials; k++) {
//            //float4 val = tex2D<float4>(partialTex, float(i * d_nPartials + (k - 1)), float(streamNo));
//            float result = 1;// partialAmp;//val.x;
//            // texPartials : the look-up table of normalized partial energy
//            sum += d_amps[k] *
//                result *
//                std::sinf(2 * Pi * d_freqs[k] * t);
//        }
//
//        d_samples[threadIdx.x * 2] = (char)sum;
//    }
//    else {
//        // block 1 compute the instrument index 50 – 99 for right channel
//            for (int k = 1; k <= d_nPartials; k++) {
//                //float4 val = tex2D<float4>(partialTex, float(i * d_nPartials + (k - 1)), float(streamNo));
//                float result = 1;// partialAmp;//val.x;
//                sum += d_amps[k] *
//                    result *
//                    std::sinf(2 * Pi * d_freqs[k] * t);
//            }
//        d_samples[threadIdx.x * 2 + 1] = (char)sum;
//    }
//}

__global__ void sinusoidal_method1(
    float* freq, // pointer to frequency array in global memory
    float* amplitude, // pointer to amplitude array in global memory
    float* samples, // pointer to sample array in global memory
    float* partialEnergy,
    int streamNo, // the stream index
    float* timeOffset // the time offset of current granularity
) {
    const float Pi = 3.141592;
    const float UnitTime = 0.01;
    float t = (float)(threadIdx.x) / sampleRate // sampleRate : 44100Hz
        + (float)streamNo * UnitTime + *timeOffset; // UnitTime : 0.01s
    float sum = 0;
    // blockIdx.x: 0 for left channel, 1 for right channel.
    if (blockIdx.x == 0) {
        // block 0 computes the instrument index 0 – 49 for left channel
        // d_nInstrument : number of instruments. (N=100)
        for (int i = 0; i < d_nInstrument / 2; i++) {
            // each instrument has 50 partials
            // d_ nPartials: number of partials of one instrument.
            for (int k = 0; k < d_nPartials; k++) {
                // texPartials : the look-up table of normalized partial energy
                sum += amplitude[i] *
                    partialEnergy[k] *
                    std::sinf(2 * Pi * freq[i] * k * t);
            }
        }
        samples[threadIdx.x * 2] = sum;
        //printf("Sum: %d", sum);
    }
    else {
        // block 1 compute the instrument index 50 – 99 for right channel
        for (int i = d_nInstrument / 2; i < d_nInstrument; i++) {
            for (int k = 1; k <= d_nPartials; k++) {
                sum += amplitude[i] *
                    partialEnergy[k] *
                    std::sinf(2 * Pi * freq[i] * k * t);
            }
        }
        samples[threadIdx.x * 2 + 1] = sum;
        //printf("Sum: %d", sum);
    }
}

//__global__ void sinusoidal_method2_1(
//    float* freq, // pointer to frequency array in global memory
//    float* amplitude, // pointer to amplitude array in global memory
//    short* samples, // pointer to sample array in global memory
//    int streamNo, // the stream index
//    float timeOffset // the time offset of current granularity
//) {
//    const float Pi = 3.141592;
//    const float UnitTime = 0.01;
//    float t = (float)(threadIdx.x) / sampleRate // sampleRate : 44100Hz
//        + (float)streamNo * UnitTime + timeOffset; // UnitTime : 0.01s
//    float sum = 0;
//    // The frequency and amplitude in one block is the same.
//    __shared__ float f;
//    __shared__ float a;
//    if(threadIdx.x == 0) {
//        f = freq[blockIdx.x];
//        a = amplitude[blockIdx.x];
//    }
//    __syncthreads();
//    // one instrument has 50 partials
//    for (int k = 1; k <= d_nPartials; k++) {
//        // texPartials : the look-up table of normalized partial energy
//        sum += a *
//            tex2D(texPartials, blockIdx.x * d_nPartials + (k - 1), streamNo)
//            * __sinf(2 * Pi * f * k * t);
//    }
//    // assign sum register to temporary storing matrix in global memory
//    stored_matrix[blockIdx.x * blockDim.x + threadIdx.x] = sum;
//}
///********** second kernel **********/
//__global__ void sinusoidal_method2_2(short* samples) {
//    float sum = 0;
//    // blockIdx.x: 0 for left channel, 1 for right channel.
//    if (blockIdx.x == 0) {
//        // block 0 compute the instrument index 0 – 49 for left channel
//        for (int i = 0; i < d_nInstrument / 2; i++)
//            sum += stored_matrix[threadIdx.x + i * blockDim.x];
//        // assign sum register to sample array in global memory
//        samples[threadIdx.x * 2] = (short)sum;
//    }
//    else {
//        // block 1 compute the instrument index 50 – 99 for right channel
//        // d_nInstrument : number of instruments. (N=100)
//        for (int i = d_nInstrument / 2; i < d_nInstrument; i++)
//            sum += stored_matrix[threadIdx.x + i * blockDim.x];
//        // assign sum register to sample array in global memory
//        samples[threadIdx.x * 2 + 1] = (short)sum;
//    }

void printError() {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        // Handle error
    }
}

void prepareSMS(int bufferSize) {
    ////move array data into to host-side arrays
    //for (int i = 0; i < d_nPartials; i++) h_freqs[i] = freqs[i];
    //for (int i = 0; i < d_nPartials; i++) h_amps[i] = amps[i];

    //allocate GPU memory
    std::cout << "CUDA Mallocs" << std::endl;
    cudaMalloc((void**) &d_freqs, sizeof(float) * d_nPartials);
    printError();
    cudaMalloc((void**) &d_amps, sizeof(float) * d_nPartials);
    printError();
    cudaMalloc((void**) &d_samples, sizeof(float) * bufferSize);
    printError();
    cudaMalloc((void**) &d_partialEnergy, sizeof(float) * table_nPartials);
    printError();


    //allocate variables for the kernel
    cudaMemcpy(d_freqs, &h_freqs, sizeof(float) * d_nPartials, cudaMemcpyHostToDevice);
    //std::cout << "memcopy d_freqs" << std::endl;
    printError();
    cudaMemcpy(d_amps, &h_amps, sizeof(float) * d_nPartials, cudaMemcpyHostToDevice);
    //std::cout << "memcopy d_amps" << std::endl;
    printError();
    //cudaMemcpy(d_samples, h_samples, sizeof(float) * bufferSize, cudaMemcpyHostToDevice);
    //std::cout << "memcopy d_samples" << std::endl;
    cudaMemcpy(d_partialEnergy, &partialEnergy, sizeof(float) * table_nPartials, cudaMemcpyHostToDevice);
    //std::cout << "memcopy d_partialEnergy" << std::endl;
    printError();
}



void shutdownSMS() {
    cudaFree(d_freqs);
    cudaFree(d_amps);
    cudaFree(d_samples);
    cudaFree(d_partialEnergy);
}

void renderBufferNewSMS(int size, float granularity, float* h_samples) {
    float* d_granularity;
    cudaMalloc((void**)&d_granularity, sizeof(float));
    cudaMemcpy(d_granularity, &granularity, sizeof(float), cudaMemcpyHostToDevice);
    sinusoidal_method1 <<<2, 441 >>> (d_freqs, d_amps, d_samples, d_partialEnergy, 0, d_granularity);
    printError();
    cudaFree(d_granularity);
    //copy memory from device to host
    cudaMemcpy(h_samples, d_samples, numSamples * sizeof(float), cudaMemcpyDeviceToHost);
}

void printSamples(float* h_samples, int size) {
    for (int i = 0; i < size; i++) {
        std::cout << h_samples[i] << ", ";
    }
}

long long calcAverage(long long samps[100]) {
    long long sum = 0;
    for (int i = 0; i < 100; i++)
        sum += samps[i];
    return sum / 100;
}

int main()
{

    //disregarding that 2D texture for now
    //set up partial energy texture object

    //all partials on full blast - the lazy way
    for (int i = 0; i < table_nPartials; i++) {
        std::cout << partialEnergy[i] << ", ";
    }

    std::cout << "Generating Audio..." << std::endl;
    //MAX THREADS 1024

    long long avg[100];
    //Time measurements
    std::cout << "Sinusoidal Method 1 Run Times" << std::endl;
    for (int i = 0; i < sizeof(granularities) / sizeof(int); i++) {
        float* h_samples = (float*)malloc(sizeof(float) * granularities[i]);
        prepareSMS(granularities[i]);
        std::cout << "Granularity: " << granularity_times[i] << std::endl;
        for (int j = 0; j < 100; j++) {

            auto t1 = std::chrono::high_resolution_clock::now();
            for (int r = 0; r < granularities[i] / 882; r++) {
                renderBufferNewSMS(granularities[i], 0.01, h_samples);
            }
            auto t2 = std::chrono::high_resolution_clock::now();
            auto us_int = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
            avg[j] = (long long)us_int.count();
        }
        std::cout << calcAverage(avg) << std::endl;
        shutdownSMS();
        free(h_samples);
    }

    //printSamples(h_samples, numSamples);

    std::cout << std::endl;

    /*for (int i = 0; i < 882; i++)
        std::cout << h_samples[i] << ", ";
    std::cout << std::endl;*/

    shutdownSMS();
   

    return 0;
}