#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/count.h>
#include <thrust/copy.h>

using namespace thrust;

struct partitionFunctor
{
	const int a;
	partitionFunctor (int _a): a(_a) {}
	__host__ __device__ bool operator()(const int x){
		if (!a)
			return (x % 2) == 0;
		else
			return (x % 2) != 0;
	}
};

template <typename T>
__global__ void meanVarianceKernel(T* partition, T* features, int partitionSize, int featuresSize, T* mean, T* variance) {

	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	mean[idx] = 0;
	variance[idx] = 0;
	for (int i = 0; i < partitionSize; ++i) {
		mean[idx] += features[partition[i] + featuresSize * idx];
	}
	mean[idx] /= partitionSize;
	T x = mean[idx];
	for (int i = 0; i < partitionSize; ++i) {
		T y = features[partition[i] + featuresSize * idx];
		variance[idx] += (y - x) * (y - x);
	}
	variance[idx] /= partitionSize;
}

template <typename T>
__global__ void probablityFeatureKernel(T* mean, T* variance, T* testData, int testDataSize, int offset, T* pfc) {

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	float product = 1.0;
	for (int i = 0; i < testDataSize; ++i) {
		int index = idx * testDataSize + i;
		product *= (1 / sqrt( 2 * 3.14 * variance[index])) * expf(-0.5 * pow((testData[index] - mean[index]), 2) / variance[index]);
	}
	pfc[idx * testDataSize + offset ] = product;
}

template <typename T>
__host__ device_vector<device_vector<T>> preProbablity(device_vector<T> deviceY) {
	device_vector<device_vector<T>> devicePreProb;
	for (auto deviceYElements : deviceY) {
		device_vector<T> tempPreProb;
		tempPreProb.push_back(count(deviceY.begin(), deviceY.end(), 0) / deviceY.size());
		tempPreProb.push_back(count(deviceY.begin(), deviceY.end(), 1) / deviceY.size());
		devicePreProb.push_back(tempPreProb);
	}
	return devicePreProb;
}

template <typename T> __host__
tuple<device_vector<device_vector<T>>, device_vector<device_vector<T>>>
meanVariance(device_vector<device_vector<T>> features, device_vector<T> deviceY, int ySize) {

	device_vector<device_vector<T>> mean, variance, partition;
	device_vector<T> tempMean(features.size());
	device_vector<T> tempVariance(features.size());
	T* ptrToFeatures = raw_pointer_cast(features.data(), features[0].size(), features.size());
	for (int i = 0; i < ySize; ++i) {
		device_vector<T> tempPart;
		copy_if(device, deviceY.begin(), deviceY.last(), tempPart.begin(), partitionFunctor(i));
		partition.push_back(tempPart);
	}
	for (auto partitionElements : partition) {
		T* ptrToPartitionElements = raw_pointer_cast(&partitionElements[0]);
		T* ptrToTempMean = raw_pointer_cast(&tempMean[0]);
		T* ptrToTempVariance = raw_pointer_cast(&tempVariance[0]);
		meanVarianceKernel <T> <<< (features.size() / 1024) + 1, (features.size() > 1024) ? 1024 : features.size() >>>
			(ptrToPartitionElements, ptrToFeatures, partitionElements.size(), features.size(), ptrToTempMean, ptrToTempVariance);
		mean.push_back(tempMean);
		variance.push_back(tempVariance);
	}
	return make_tuple(mean, variance);
}

template <typename T> __host__ device_vector<device_vector<T>> probablityFeature
(device_vector<device_vector<T>> mean, device_vector<device_vector<T>> variance,
	device_vector<device_vector<T>> testData) {

	device_vector<device_vector<T>> probabilityFeatureClass(mean.size(), mean[0].size());
	T* ptrToMean = raw_pointer_cast(mean.data(), mean[0].size(), mean.size());
	T* ptrToVariance = raw_pointer_cast(variance.data(), variance[0].size(), variance.size());
	T* ptrToTestData = raw_pointer_cast(testData.data(), testData[0].size(), testData.size());
	T* ptrToPFC = raw_pointer_cast(probabilityFeatureClass.data(), probabilityFeatureClass[0].size(), testData.size());
	for (int i = 0; i < mean.size(); ++i) {
		probablityFeatureKernel <T> <<< testData.size() / 1024 + 1, (testData.size() > 1024) ? 1024 : testData.size() >>>
			(ptrToMean, ptrToVariance, ptrToTestData, testData[0].size(), i, ptrToPFC);
	}
	return probabilityFeatureClass;
}

template <typename T> __host__ device_vector<T> gaussianNaiveBayes
(device_vector<device_vector<T>> xTrain, device_vector<T> yTrain, device_vector<device_vector<T>> testData) {

	device_vector<device_vector<T>> pcf;
	device_vector<T> totalProb;
	device_vector<T> prediction;
	auto meanVarianceValue = meanVariance(xTrain, yTrain, yTrain.size());
	auto mean = get<0>(meanVarianceValue);
	auto variance = get<1>(meanVarianceValue);
	auto pfc = probablityFeature(mean, variance, testData);
	auto preProb = preProbablity(yTrain);
	for (int i = 0; i < testData.size(); ++i) {
		auto tempProb = 0.0;
		for (int j = 0; j < mean.size(); j++) {
			tempProb += pfc[i][j] * preProb[i][j];
		}
		totalProb.push_back(tempProb);
	}
	for (int i = 0; i < testData.size(); ++i) {
		device_vector<T> tempPCF;
		for (int j = 0; j < mean.size(); j++) {
			tempPCF.push_back((pfc[i][j] * preProb[i][j]) / totalProb[i]);
		}
		pcf.push_back(tempPCF);
	}
	for (int i = 0; i < testData.size(); ++i) {
		prediction.push_back((pcf[i][0] > pcf[i][1]) ? pcf[i][0] : pcf[i][1]);
	}
	return prediction;
}