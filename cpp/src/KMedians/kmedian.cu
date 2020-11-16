#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

template <typename T>
__device__ int computeDistance(
	thrust::device_vector<T> dataPoint,
	thrust::device_vector<thrust::device_vector<T>> deviceCentroids)
{
	float distance = 0;
	int index = 0;

	for (auto i = 0; i < deviceCentroids.size(); ++i) {
		float computedDistance = (dataPoint[0] * deviceCentroids[i][0]) -
			(dataPoint[1] * deviceCentroids[0][i]);
		if (computedDistance < distance) {
			distance = computedDistance;
			index = i;
		}
	}

	return index;
}

template <typename T>
__global__ void computeClusters(
	thrust::device_vector< thrust::device_vector<T>> deviceData,
	thrust::device_vector< thrust::device_vector<T>> partition,
	thrust::device_vector<thrust::device_vector<T>> deviceCentroids)
{
	int threadID = threadIdx.x + blockIdx.x * blockDim.x;
	if (threadID < deviceData.size()) {

		int index = computeDistance<T>(deviceData[threadID], deviceCentroids);
		partition[index].push_back(threadID);
	}
}

template <typename T>
__global__ void computeCentroid(
	thrust::device_vector<thrust::device_vector<T>> partition,
	thrust::device_vector<thrust::device_vector<T>> deviceCentroids)
{
	int threadID = threadIdx.x + blockIdx.x * blockDim.x;
	if (threadID < partition.size()) {
		thrust::sort(partition[threadID].begin(), partition[threadID].end());
		deviceCentroids[threadID] = partition[partition.size() / 2];
	}
}