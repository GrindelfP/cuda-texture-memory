#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cstdio>
#include <chrono>

#define BLOCK_SIZE 16 // Размер блока для GPU
#define PI 3.14159265359

// GPU ядро для глобальной памяти
__global__ void Conv_Glb(float* dConv, float* dS, int W, int H, int delta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + delta;
    int idy = blockIdx.y * blockDim.y + threadIdx.y + delta;

    float norm = 0.0f, cov = 0.0f;

    for (int ix = -delta; ix <= delta; ix++) {
        for (int iy = -delta; iy <= delta; iy++) {
            float K = expf(-(ix * ix + iy * iy) / (delta * delta));
            cov += K * dS[idx + ix + (idy + iy) * (W + BLOCK_SIZE)];
            norm += K;
        }
    }
    dConv[idx + idy * (W + BLOCK_SIZE)] = cov / norm;
}

// GPU ядро для текстурной памяти (Texture Object API)
__global__ void Conv_Tex(float* dConv, cudaTextureObject_t texObj, int W, int H, int delta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + delta;
    int idy = blockIdx.y * blockDim.y + threadIdx.y + delta;

    float norm = 0.0f, cov = 0.0f;

    for (int ix = -delta; ix <= delta; ix++) {
        for (int iy = -delta; iy <= delta; iy++) {
            float K = expf(-(ix * ix + iy * iy) / (delta * delta));
            cov += K * tex2D<float>(texObj, idx + ix, idy + iy);
            norm += K;
        }
    }
    dConv[idx + idy * (W + BLOCK_SIZE)] = cov / norm;
}

int main() {
    int ws[] = { 1024, 5120, 10240 }, hs[] = { 2048, 5120, 10240 };

    for (int i = 0; i < 3; ++i) {

        int W = ws[i], H = hs[i];

        int delta = BLOCK_SIZE / 2, size = (W + BLOCK_SIZE) * (H + BLOCK_SIZE);
        size_t mem_size = sizeof(float) * size;

        float* hS = (float*)malloc(mem_size);
        float* hConv = (float*)malloc(mem_size);
        float* hdConv = (float*)malloc(mem_size);
        float* hdConvText = (float*)malloc(mem_size);

        for (int y = 0; y < H + BLOCK_SIZE; y++) {
            for (int x = 0; x < W + BLOCK_SIZE; x++) {
                hS[x + y * (W + BLOCK_SIZE)] = sinf(x * 2.0f * PI / (W + BLOCK_SIZE)) * sinf(y * 2.0f * PI / (H + BLOCK_SIZE));
            }
        }

        dim3 nThreads(BLOCK_SIZE, BLOCK_SIZE);
        dim3 nBlocks(W / BLOCK_SIZE, H / BLOCK_SIZE);

        // --------------------- CPU ---------------------
        auto startCPU = std::chrono::high_resolution_clock::now();
        for (int y = delta; y < H + delta; y++) {
            for (int x = delta; x < W + delta; x++) {
                float norm = 0.0f, cov = 0.0f;
                for (int iy = -delta; iy <= delta; iy++) {
                    for (int ix = -delta; ix <= delta; ix++) {
                        float K = expf(-(ix * ix + iy * iy) / (delta * delta));
                        cov += K * hS[(x + ix) + (y + iy) * (W + BLOCK_SIZE)];
                        norm += K;
                    }
                }
                hConv[x + y * (W + BLOCK_SIZE)] = cov / norm;
            }
        }
        auto endCPU = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> cpuTime = endCPU - startCPU;

        // --------------------- GPU ---------------------
        float* dS, * dConv, * dConvText;

        cudaMalloc((void**)&dS, mem_size);
        cudaMalloc((void**)&dConv, mem_size);

        cudaEvent_t start, stop;
        float timerGlobal, timerTexture;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // --------------------- GPU-GLOBAL ---------------------
        cudaMemcpy(dS, hS, mem_size, cudaMemcpyHostToDevice);
        cudaMemset(dConv, 0, mem_size);

        cudaEventRecord(start,0);

        Conv_Glb << <nBlocks, nThreads >> > (dConv, dS, W, H, delta);
        cudaDeviceSynchronize();

        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&timerGlobal, start, stop);

        cudaMemcpy(hdConv, dConv, mem_size, cudaMemcpyDeviceToHost);

        // --------------------- GPU-TEXTURE ---------------------
        cudaMalloc((void**)&dConvText, mem_size);
        cudaMemset(dConvText, 0, mem_size);

        cudaArray* cuArray;
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
        cudaMallocArray(&cuArray, &channelDesc, W + BLOCK_SIZE, H + BLOCK_SIZE);
        cudaMemcpyToArray(cuArray, 0, 0, hS, mem_size, cudaMemcpyHostToDevice);

        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray;

        cudaTextureDesc texDesc = {};
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModePoint;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = false;


        cudaTextureObject_t texObj = 0;
        cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);

        cudaEventRecord(start,0);

        Conv_Tex << <nBlocks, nThreads >> > (dConvText, texObj, W, H, delta);
        cudaDeviceSynchronize();

        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&timerTexture, start, stop);

        cudaMemcpy(hdConvText, dConv, mem_size, cudaMemcpyDeviceToHost);

        std::cout << "CPU time: " << cpuTime.count() << " ms\n";
        std::cout << "GPU (Global memory) time: " << timerGlobal << " ms\n";
        std::cout << "GPU (Texture memory) time: " << timerTexture << " ms\n";
        std::cout << "Speedup (Global): " << cpuTime.count() / timerGlobal << "x\n";
        std::cout << "Speedup (Texture): " << cpuTime.count() / timerTexture << "x\n";

        cudaFree(dS);
        cudaFree(dConv);
        cudaFreeArray(cuArray);
        cudaDestroyTextureObject(texObj);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        free(hS);
        free(hConv);
        free(hdConv);
    }

    return 0;
}
