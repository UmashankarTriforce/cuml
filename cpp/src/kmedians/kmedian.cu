#include <cuda.h>
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

	for (auto i = 0; i < deviceCentriods.size(); ++i) {
		float computedDistance = (dataPoint[0] * deviceCentroids[i][0]) - \
			(dataPoint[1] * deviceCentroids[0][i];
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
		thrust::device_vector<T>::iterator itr = thrust::find(deviceData.begin(), 
			deviceData.end(), deviceData[index]);
		partition[index].push_back(thrust::distance(deviceData.begin(), itr));
	}
}

template <typename T>
__global__ void computeCentroid(
	thrust::device_vector<thrust::device_vector<T>> deviceData,
	thrust::device_vector<thrust::device_vector<T>> partition,
	thrust::device_vector<thrust::device_vector<T>> deviceCentroids)
{
	int threadID = threadID.x + blockIdx.x * blockDim.x;
	if (threadID < partition.size()) {
		for (int i = 0; i < partition.size(); ++i) {
			thrust::sort(partition[i].begin(), partition[i].end());
			deviceCentroids[i] = partition[partition.size() / 2];
		}
	}
}


template <typename T>
__host__ thrust::host_vector<thrust::host_vector<T>>
computeKMedians(thrust::host_vector<thrust::host_vector<T>> hostData, 
	int clusters, int iterations) 
{
	
	thrust::device_vector<thrust::device_vector<T>> deviceData = hostData;
	thrust::device_vector<thrust::device_vector<int>> partition;
	thrust::device_vector<thrust::device_vector<T>> deviceCentroids;
	thrust::host_vector<thrust::host_vector<T>> hostCentroids;
	
	//choosing the initial centroids
	
	for (auto i = 0; i < clusters; ++i) {
		
		deviceCentroids.push_back(deviceData[rand() % deviceData.size()]);
		partition.push_back(thrust::device_vector<T>());

	}

	// computing clusters

	for (auto i = 0; i < iterations; ++i) {
		computeClusters<T> <<< deviceData.size() / 1024 + 1, 1024 >>> (deviceData,
			partition, deviceCentroids);
		cudaDeviceSynchronize();
		computeCentroid<T> <<< partition.size() / 1024 + 1, 1024 >>> (deviceData,
			partition, deviceCentroids);
		cudaDeviceSynchronize();
	}

	hostCentroids = deviceCentroids;
	return hostCentroids;
}
