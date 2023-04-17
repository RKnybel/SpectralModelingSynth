
#define NOMINMAX
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "olcNoiseMaker.h"

#include <stdio.h>
#include <string.h>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <queue>
#include <Windows.h>
#include <mmsystem.h>

std::queue<char*> audioBufferQueue; // queue to hold audio buffers

const int sampleRate = 44100;
const int d_nInstrument = 100; // can also think of instruments as paraphonic oscillators
const int d_nPartials = 50;

//partial frequency array. Each of the 50 sine wave partials gets a frequency
float freqs[50] = { 1000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000 };
float* h_freqs = (float*)malloc(sizeof(float) * 50);
float* d_freqs;

//partial amplitude array. Each of the 50 sine wave partials gets an amplitude
float amps[50] = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
float* h_amps = (float*)malloc(sizeof(float) * 50);
float* d_amps;
int numSamples = 882;

char* h_samples = (char*)malloc(sizeof(char) * numSamples);
char* d_samples;

//partial table. I want to learn what this does.
int table_nInstruments = 50;
int table_nPartials = 50;
float partialEnergy[50];
float* d_partialEnergy;
float* h_partialArray = (float*)malloc(sizeof(float)* table_nInstruments* table_nPartials);
float* d_partialArray;
cudaTextureObject_t texPartials;

//sount fonts? samples? Arrays of normalized partial data to use for textures:
// 


// new algorithm (3?) - 
__global__ void new_method(
    float* d_freqs, // pointer to frequency array in global memory
    float* d_amps, // pointer to amplitude array in global memory
    char* d_samples, // pointer to sample array in global memory
    int streamNo, // the stream index
    float timeOffset // the time offset of current granularity
) {
    const float Pi = 3.141592;
    const float UnitTime = 0.01;
    float t = (float)(threadIdx.x) / sampleRate // sampleRate : 44100Hz
        + (float)streamNo * UnitTime + timeOffset; // UnitTime : 0.01s
    float sum = 0;
    float partialAmp = 1.0 / d_nPartials;
    // blockIdx.x: 0 for left channel, 1 for right channel.
    if (blockIdx.x == 0) {
        // block 0 computes the instrument index 0 – 49 for left channel
        // d_nInstrument : number of instruments. (N=100)
        for (int k = 0; k < d_nPartials; k++) {
            //float4 val = tex2D<float4>(partialTex, float(i * d_nPartials + (k - 1)), float(streamNo));
            float result = 1;// partialAmp;//val.x;
            // texPartials : the look-up table of normalized partial energy
            sum += d_amps[k] *
                result *
                std::sinf(2 * Pi * d_freqs[k] * t);
        }

        d_samples[threadIdx.x * 2] = (char)sum;
    }
    else {
        // block 1 compute the instrument index 50 – 99 for right channel
            for (int k = 1; k <= d_nPartials; k++) {
                //float4 val = tex2D<float4>(partialTex, float(i * d_nPartials + (k - 1)), float(streamNo));
                float result = 1;// partialAmp;//val.x;
                sum += d_amps[k] *
                    result *
                    std::sinf(2 * Pi * d_freqs[k] * t);
            }
        d_samples[threadIdx.x * 2 + 1] = (char)sum;
    }
}

__global__ void sinusoidal_mehod1(
    float* freq, // pointer to frequency array in global memory
    float* amplitude, // pointer to amplitude array in global memory
    char* samples, // pointer to sample array in global memory
    float* partialEnergy,
    int streamNo, // the stream index
    float timeOffset // the time offset of current granularity
) {
    const float Pi = 3.141592;
    const float UnitTime = 0.01;
    float t = (float)(threadIdx.x) / sampleRate // sampleRate : 44100Hz
        + (float)streamNo * UnitTime + timeOffset; // UnitTime : 0.01s
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
        samples[threadIdx.x * 2] = (char)sum;
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
        samples[threadIdx.x * 2 + 1] = (char)sum;
    }
}


void CALLBACK waveOutProc(HWAVEOUT hwo, UINT uMsg, DWORD_PTR dwInstance, DWORD_PTR dwParam1, DWORD_PTR dwParam2) {
    if (uMsg == 955) {
        if (!audioBufferQueue.empty()) {
            WAVEHDR waveHdr;
            waveHdr.lpData = audioBufferQueue.front();
            waveHdr.dwBufferLength = 441 * sizeof(short) * 2;
            waveHdr.dwBytesRecorded = 0;
            waveHdr.dwUser = 0;
            waveHdr.dwFlags = 0;
            waveHdr.dwLoops = 0;
            int result = waveOutPrepareHeader(hwo, &waveHdr, sizeof(WAVEHDR));
            if (result != MMSYSERR_NOERROR) //err 12 handle being used
            {
                // Failed to prepare the audio buffer header
                std::cout << result;
                waveOutClose(hwo);
            }
            free(audioBufferQueue.front());
            audioBufferQueue.pop();
        }
    }
};

void printError() {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        // Handle error
    }
}

void prepareSMS() {
    //move array data into to host-side arrays
    for (int i = 0; i < d_nPartials; i++) h_freqs[i] = freqs[i];
    for (int i = 0; i < d_nPartials; i++) h_amps[i] = amps[i];

    //allocate GPU memory
    std::cout << "CUDA Mallocs" << std::endl;
    cudaMalloc((void**) & d_freqs, sizeof(float) * d_nPartials);
    printError();
    cudaMalloc((void**) &d_amps, sizeof(float) * d_nPartials);
    printError();
    cudaMalloc((void**) &d_samples, sizeof(char) * numSamples);
    printError();
    cudaMalloc((void**)&d_partialEnergy, sizeof(float) * table_nPartials);
    printError();


    //allocate variables for the kernel
    cudaMemcpy(d_freqs, h_freqs, sizeof(float) * d_nPartials, cudaMemcpyHostToDevice);
    std::cout << "memcopy d_freqs" << std::endl;
    printError();
    cudaMemcpy(d_amps, h_amps, sizeof(float) * d_nPartials, cudaMemcpyHostToDevice);
    std::cout << "memcopy d_amps" << std::endl;
    printError();
    cudaMemcpy(d_samples, h_samples, sizeof(char) * 882, cudaMemcpyHostToDevice);
    std::cout << "memcopy d_samples" << std::endl;
    cudaMemcpy(d_partialEnergy, &partialEnergy, sizeof(float) * table_nPartials, cudaMemcpyHostToDevice);
    std::cout << "memcopy d_partialEnergy" << std::endl;
    printError();
}

void shutdownSMS() {
    //free memory
    cudaFree(d_freqs);
    cudaFree(d_amps);
    cudaFree(d_samples);
    cudaFree(d_partialArray);
    free(h_freqs);
    free(h_amps);
    free(h_samples);
    free(h_partialArray);
}

char* renderBufferNewSMS() {
    char* new_h_samples = (char*)malloc(sizeof(char) * numSamples);
    sinusoidal_mehod1 <<<2, 441 >>> (d_freqs, d_amps, d_samples, d_partialEnergy, 0, 0.01);
    //copy memory from device to host
    cudaMemcpy(new_h_samples, d_samples, numSamples * sizeof(char), cudaMemcpyDeviceToHost);
    return new_h_samples;
}

// Generate and add audio buffer to the queue
void generateAudioBufferAndAddToQueue()
{
    char* buffer = renderBufferNewSMS();
    audioBufferQueue.push(buffer);
}

int playBuffer(char* buffer, int bufferSize) {
    //audio player code
     // Open the default waveform-audio output device
    HWAVEOUT hWaveOut;
    WAVEFORMATEX format;
    format.wFormatTag = WAVE_FORMAT_PCM;
    format.nChannels = 2;
    format.nSamplesPerSec = 44100;
    format.wBitsPerSample = 16;
    format.nBlockAlign = format.nChannels * format.wBitsPerSample / 8;
    format.nAvgBytesPerSec = format.nSamplesPerSec * format.nBlockAlign;
    format.cbSize = 0;
    MMRESULT result = waveOutOpen(&hWaveOut, WAVE_MAPPER, &format, (DWORD_PTR)waveOutProc, 0, CALLBACK_FUNCTION);
    if (result != MMSYSERR_NOERROR)
    {
        // Failed to open the audio device
        return 1;
    }

    // Prepare the audio buffer header
    WAVEHDR waveHdr;
    waveHdr.lpData = (char*)buffer;
    waveHdr.dwBufferLength = bufferSize * sizeof(short) * format.nChannels;
    waveHdr.dwBytesRecorded = 0;
    waveHdr.dwUser = 0;
    waveHdr.dwFlags = 0;
    waveHdr.dwLoops = 0;
    result = waveOutPrepareHeader(hWaveOut, &waveHdr, sizeof(WAVEHDR));
    if (result != MMSYSERR_NOERROR)
    {
        // Failed to prepare the audio buffer header
        waveOutClose(hWaveOut);
        return 1;
    }

    // Play the audio buffer
    result = waveOutWrite(hWaveOut, &waveHdr, sizeof(WAVEHDR));
    if (result != MMSYSERR_NOERROR)
    {
        // Failed to start audio playback
        waveOutUnprepareHeader(hWaveOut, &waveHdr, sizeof(WAVEHDR));
        waveOutClose(hWaveOut);
        return 1;
    }

    // Wait for the audio buffer to finish playing
    while (waveHdr.dwFlags & WHDR_DONE) {
        Sleep(1);
    }

    // Clean up
    waveOutUnprepareHeader(hWaveOut, &waveHdr, sizeof(WAVEHDR));
    waveOutClose(hWaveOut);
    // delete[] audioBuffer;
    return 0;
}

//Play audio buffers in the queue
void playAudioBuffersInQueue()
{
    int buffNum = 1;
    while (!audioBufferQueue.empty()) {
        playBuffer(audioBufferQueue.front(), numSamples/2);
        free(audioBufferQueue.front());
        audioBufferQueue.pop();
        //std::cout << "Buffer number " << buffNum << std::endl;
        buffNum++;
    }
}

double MakeNoise(double dTime) {
    return 0;
}


int main()
{

    prepareSMS();

    //disregarding that 2D texture for now
    //set up partial energy texture object

    //all partials on full blast - the lazy way
    for (int i = 0; i < table_nInstruments; i++) {
            partialEnergy[i] = 1.0;
    }

    std::cout << "Generating Audio..." << std::endl;

    for (int i = 0; i < 100; i++) {
        generateAudioBufferAndAddToQueue();
    }

    for (int i = 0; i < numSamples; i++) {
        std::cout << (int)h_samples[i] << ", ";
    }

    std::cout << std::endl;

    /*for (int i = 0; i < 882; i++)
        std::cout << h_samples[i] << ", ";
    std::cout << std::endl;*/

    shutdownSMS();
   

    return 0;
}