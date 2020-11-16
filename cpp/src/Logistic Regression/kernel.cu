#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"

#define REDUCE_BLOCK_SIZE 128

__global__ void matrixMulKernel(float* m1, float* m2, float* r, int m1w, int m2w, int rw, int rh)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if ((row < rh) && (col < rw)) {
		// dot product
		float accum = 0.0f;
		for (int c = 0; c < m1w; c++)
		{
			float v1 = m1[row * m1w + c];
			float v2 = m2[c * m2w + col];
			accum += (v1 * v2);
		}

		r[row * rw + col] = accum;
	}
}

__global__ void sigmoidKernel(float* r, int m)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < m) {
		float val = r[index];
		r[index] = 1.0 / (1.0 + expf(-val));
	}
}

__global__ void matrixAbsErrorKernel(float* p, float* ys, float* r, int rw, int rh)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if ((row < rh) && (col < rw)) {
		float pval = p[row * rw + col];
		float ysval = ys[row * rw + col];

		float v = pval - ysval;
		r[row * rw + col] = v * v;
	}
}

__global__ void absErrorKernel(float* p, float* ys, float* r, int m)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < m) {
		float pval = p[index];
		float ysval = ys[index];

		float v = pval - ysval;
		r[index] = v * v;
	}
}

__global__ void updateParamsAbsErrorKernel(float* p, float* ys, float* th, float* xs, int m, float alpha)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < m) {
		float h = *p;
		float y = *ys;

		float x = xs[index];

		th[index] = th[index] - alpha * (h - y) * x;
	}
}

__global__ void crossEntropyKernel(float* p, float* ys, float* r, int m)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < m) {
		float pval = p[index];
		float ysval = ys[index];

		float ex = log1pf(expf(-ysval * pval));
		r[index] = ex;
	}
}

__global__ void reduceKernel(float* input, float* output, int len) {
	//@@ Load a segment of the input vector into shared memory
	__shared__ float partialSum[2 * REDUCE_BLOCK_SIZE];
	unsigned int t = threadIdx.x, start = 2 * blockIdx.x * REDUCE_BLOCK_SIZE;
	if (start + t < len)
		partialSum[t] = input[start + t];
	else
		partialSum[t] = 0;
	if (start + REDUCE_BLOCK_SIZE + t < len)
		partialSum[REDUCE_BLOCK_SIZE + t] = input[start + REDUCE_BLOCK_SIZE + t];
	else
		partialSum[REDUCE_BLOCK_SIZE + t] = 0;
	//@@ Traverse the reduction tree
	for (unsigned int stride = REDUCE_BLOCK_SIZE; stride >= 1; stride >>= 1) {
		__syncthreads();
		if (t < stride)
			partialSum[t] += partialSum[t + stride];
	}
	//@@ Write the computed sum of the block to the output vector at the
	//@@ correct index
	if (t == 0)
		output[blockIdx.x] = partialSum[0];
}