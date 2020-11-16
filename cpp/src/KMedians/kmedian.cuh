#include <ctime>
#include <iostream>

#include "kmedian.cu"

using namespace thrust;
using namespace std;

template <typename T>
class KMedians {

private:

	// host vectors
	host_vector<host_vector<T>> hostArr;
	host_vector<host_vector<T>> hostPartition;
	//device vectors
	device_vector<device_vector<T>> deviceArr;
	device_vector<device_vector<T>> deviceCentroids;
	device_vector<device_vector<T>> devicePartition;

	// host functions
	device_vector<device_vector<T>> initPartition(int cluster);

public:

	// host functions
	KMedians(host_vector<host_vector<T>> host);
	void fit(int iterations, int clusters);
	void predict(host_vector<host_vector<T>> host);

};

//public functions

template <typename T> KMedians<T>::KMedians(host_vector<host_vector<T>> host) {
	hostArr = host;
}

template <typename T> void KMedians<T>::fit(int iterations, int clusters) {

	deviceArr = hostArr;

	//choosing the initial centroids

	for (auto i = 0; i < clusters; ++i) {
		deviceCentroids.push_back(deviceArr[rand() % deviceArr.size()]);
	}

	for (auto i = 0; i < iterations; ++i) {
		devicePartition = initPartition(clusters);
		int threads = (deviceArr.size() > 1024) ? 1024 : deviceArr.size();
		int blocks = (deviceArr.size() > 1024) ?
			deviceArr.size() / 1024 + 1 : 1;
		computeClusters<T> <<< blocks, threads >>>
			(deviceArr, devicePartition, deviceCentroids);
		cudaDeviceSynchronize();
		threads = (devicePartition.size() > 1024) ?
			1024 : devicePartition.size();
		blocks = (devicePartition.size() > 1024) ?
			devicePartition.size() / 1024 + 1 : 1;
		computeCentroid<T> <<< blocks, threads >>>
			(devicePartition, deviceCentroids);
	}

}

// private host functions

template <typename T> device_vector<device_vector<T>> KMedians<T>::
initPartition(int cluster) {

	device_vector<device_vector<T>> partition;
	for (auto i = 0; i < cluster; ++i) {
		partition.push_back(device_vector<T>());
	}
	return partition;
}