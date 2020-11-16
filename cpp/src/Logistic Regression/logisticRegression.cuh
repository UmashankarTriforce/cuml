#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <numeric>
#include <stdlib.h>
#include "kernel.cu"
#include <vector>
#include <algorithm>
using namespace std;


struct Matrix {
	Matrix() : elements(NULL), width(0), height(0), pitch(0) {}
	~Matrix() { if (elements) delete[] elements; }
	unsigned int width;
	unsigned int height;
	unsigned int pitch;
	float* elements;
};

#define SAFE_CALL(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)


inline static void InitializeMatrix(Matrix* mat, int x, int y, float val) {
	if (x > mat->width || y > mat->height) {
		throw ("invalid access - Initialize Matrix");
	}
	mat->elements[y * mat->width + x] = val;
}

inline static float Matrix_Element_Required(Matrix* mat, int x, int y)
{
	if (x > mat->width || y > mat->height) {
		throw ("invalid access - Matrix Element Required");
	}
	return mat->elements[y * mat->width + x];
}

static void AllocateMatrix(Matrix* mat, int height, int width)
{
	mat->elements = new float[height * width];
	mat->width = width;
	mat->height = height;
	for (int i = 0; i < mat->width; i++) {
		for (int j = 0; j < mat->height; j++) {
			InitializeMatrix(mat, i, j, 0.0f);
		}
	}
}

static void DisplayMatrix(Matrix& mat, bool force = false)
{
	std::cout << "Dim: " << mat.height << ", " << mat.width << "\n";
	if ((mat.width < 10 && mat.height < 10) || force)
	{
		for (int j = 0; j < mat.height; j++) {
			for (int i = 0; i < mat.width; i++) {
				std::cout << Matrix_Element_Required(&mat, i, j) << "\t";
			}
			std::cout << "\n";
		}
	}
	std::cout << std::endl;
}

static bool setup_data(string file_name, Matrix* X, Matrix* y) {

	ifstream s(file_name.c_str());
	//ifstream s(file_name);
	if (!s.is_open()) {
		//throw runtime_error(file_name + " doesn't exist");
		printf("The file does not exist\n");
	}

	int rows = 0;
	int cols = 0;
	string line;
	while (getline(s, line)) {
		// if we read first line, check how many columns
		if (rows++ == 0) {
			stringstream ss(line);

			while (ss.good()) {
				string substr;
				getline(ss, substr, ',');
				cols++;
			}
		}
	}
	std::cout << "Found " << rows << " rows with " << cols << " columns." << std::endl;
	s.clear();
	s.seekg(0, ios::beg);

	AllocateMatrix(X, rows - 1, cols - 2);
	AllocateMatrix(y, rows - 1, 1);

	// go to second line
	getline(s, line);
	int ya = 0;
	while (getline(s, line)) {
		stringstream ss(line);

		int xa = 0;
		while (ss.good()) {
			string substr;
			getline(ss, substr, ',');
			// first column is uninteresting
			// second column is target values
			if (xa == 1) {
				float val = atof(substr.c_str());
				InitializeMatrix(y, 0, ya, val);
			}
			else if (xa > 1) {
				float val = atof(substr.c_str());
				InitializeMatrix(X, (xa - 2), ya, val);
			}
			xa++;
		}
		ya++;
	}

	return true;
}

static void Normalize_Matrix_min_max(Matrix* m)
{
	for (int x = 0; x < m->width; ++x) {
		// calculate std for each column
		float min = Matrix_Element_Required(m, x, 0);
		float max = Matrix_Element_Required(m, x, 0);
		for (int y = 1; y < m->height; ++y) {
			float val = Matrix_Element_Required(m, x, y);
			if (val < min) {
				min = val;
			}
			else if (val > max) {
				max = val;
			}
		}

		for (int y = 0; y < m->height; ++y) {
			float val = Matrix_Element_Required(m, x, y);
			InitializeMatrix(m, x, y, (val - min) / max);
		}
	}
}

static void InitializeRandom(Matrix* mat, float LO, float HI)
{
	for (int i = 0; i < mat->width; ++i) {
		for (int j = 0; j < mat->height; ++j) {
			float r = LO + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (HI - LO)));
			InitializeMatrix(mat, i, j, r);
		}
	}
}

static void CheckCudaErrorAux(const char* file, unsigned line, const char* statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement << " returned " << cudaGetErrorString(err) << "(" << err << ") at " << file << ":" << line << std::endl;
	exit(1);
}


static void Logistic_Regression_CUDA(Matrix* X, Matrix* y, Matrix* Parameters, Matrix* Train_Parameters, int maxIterations, float alpha, vector<float>& cost_function)
{
	// put stuff into gpu
	float* gpu_X;
	float* gpu_y;

	float* gpu_prediction;

	float* gpu_params;
	float* gpu_abs_error;
	float* gpu_err_cost;

	float* gpu_predictions;
	Matrix predictions;
	AllocateMatrix(&predictions, y->height, y->width);

	Matrix absErrors;
	AllocateMatrix(&absErrors, y->height, y->width);

	float mean_error;
	float sum = 0;
	int quantity = 1;

	int m = y->height;

	int numOutputElements;
	numOutputElements = m / (REDUCE_BLOCK_SIZE << 1);
	if (m % (REDUCE_BLOCK_SIZE << 1)) {
		numOutputElements++;
	}

	SAFE_CALL(cudaMalloc((void**)&gpu_X, sizeof(float) * X->width * X->height));
	SAFE_CALL(cudaMalloc((void**)&gpu_y, sizeof(float) * y->width * y->height));
	SAFE_CALL(cudaMalloc((void**)&gpu_prediction, sizeof(float)));
	SAFE_CALL(cudaMalloc((void**)&gpu_predictions, sizeof(float) * y->width * y->height));
	SAFE_CALL(cudaMalloc((void**)&gpu_abs_error, sizeof(float) * y->width * y->height));
	SAFE_CALL(cudaMalloc((void**)&gpu_params, sizeof(float) * Parameters->width * Parameters->height));
	SAFE_CALL(cudaMalloc((void**)&gpu_err_cost, sizeof(float) * numOutputElements));

	SAFE_CALL(cudaMemcpy(gpu_X, X->elements, sizeof(float) * X->width * X->height, cudaMemcpyHostToDevice));
	SAFE_CALL(cudaMemcpy(gpu_y, y->elements, sizeof(float) * y->width * y->height, cudaMemcpyHostToDevice));
	SAFE_CALL(cudaMemcpy(gpu_params, Parameters->elements, sizeof(float) * Parameters->width * Parameters->height, cudaMemcpyHostToDevice));

	// invoke kernel
	static const int blockWidth = 16;
	static const int blockHeight = blockWidth;
	int numBlocksW = X->width / blockWidth;
	int numBlocksH = X->height / blockHeight;
	if (X->width % blockWidth) numBlocksW++;
	if (X->height % blockHeight) numBlocksH++;

	dim3 dimGrid(numBlocksW, numBlocksH);
	dim3 dimBlock(blockWidth, blockHeight);

	dim3 dimReduce((m - 1) / REDUCE_BLOCK_SIZE + 1);
	dim3 dimReduceBlock(REDUCE_BLOCK_SIZE);

	dim3 dimVectorGrid(((m - 1) / blockWidth * blockWidth) + 1);
	dim3 dimVectorBlock(blockWidth * blockWidth);

	float* error_accum = new float[numOutputElements];
	for (int iter = 0; iter < maxIterations; ++iter) {
		for (int i = 0; i < m; ++i) {
			matrixMulKernel <<<dimGrid, dimBlock >>> (&gpu_X[i * X->width], gpu_params, gpu_prediction, X->width, Parameters->width, 1, 1);
			sigmoidKernel <<<dimVectorGrid, dimVectorBlock >>> (gpu_prediction, 1);
			updateParamsAbsErrorKernel <<<dimVectorGrid, dimVectorBlock >>> (gpu_prediction, &gpu_y[i], gpu_params, &gpu_X[i * X->width], Parameters->height, alpha);
		}
		matrixMulKernel <<<dimGrid, dimBlock >>> (gpu_X, gpu_params, gpu_predictions, X->width, Parameters->width, predictions.width, predictions.height);
		sigmoidKernel <<<dimVectorGrid, dimVectorBlock >>> (gpu_predictions, m);


		// calculate error
		absErrorKernel <<<dimVectorGrid, dimVectorBlock >>> (gpu_predictions, gpu_y, gpu_abs_error, m);
		reduceKernel <<<dimReduce, dimReduceBlock >>> (gpu_abs_error, gpu_err_cost, m);
		SAFE_CALL(cudaMemcpy(error_accum, gpu_err_cost, sizeof(float) * numOutputElements, cudaMemcpyDeviceToHost));
		float g_sum = 0;
		for (int i = 0; i < numOutputElements; ++i)
		{
			g_sum += error_accum[i];
		}

		g_sum /= (2 * m);

		cost_function.push_back(g_sum);
		sum += g_sum;
		quantity++;
		cout << g_sum << "\n";
	}

	mean_error = sum / quantity;
	printf("\n The mean error is %f\n", mean_error);
	cout << endl;

	delete[] error_accum;
	SAFE_CALL(cudaFree(gpu_X));
	SAFE_CALL(cudaFree(gpu_y));
	SAFE_CALL(cudaFree(gpu_abs_error));
	SAFE_CALL(cudaFree(gpu_prediction));
	SAFE_CALL(cudaFree(gpu_predictions));
	SAFE_CALL(cudaFree(gpu_params));
	SAFE_CALL(cudaFree(gpu_err_cost));
}
