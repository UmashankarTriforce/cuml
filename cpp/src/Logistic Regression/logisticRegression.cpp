#include "logisticRegression.cuh"

int main(int argc, char* argv[])
{
    string input_file = "";
    cout << "Please enter a valid file to run test for logistic regression on CUDA:\n>";
    getline(cin, input_file);
    cout << "You entered: " << input_file << endl << endl;
    Matrix X, y;
    setup_data(input_file, &X, &y);
    cout << "\n The X - Squiggle Matrix." << endl;
    DisplayMatrix(X, true);
    cout << "\n The y - Matrix." << endl;
    DisplayMatrix(y, true);

    Matrix Parameters, Train_Parameters;
    //Setup matrices with 1 as value initially
    AllocateMatrix(&Parameters, X.width, 1);
    AllocateMatrix(&Train_Parameters, X.width, 1);
    //Initialize with random +1 and -1 parameters.
    InitializeRandom(&Parameters, -1.0, 1.0);

    Normalize_Matrix_min_max(&X);

    vector<float> cost_function;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //unsigned int timer;
    //CUT_SAFE_CALL(cutCreateTimer(&timer));

    //cutStartTimer(timer);
    cudaEventRecord(start);
    Logistic_Regression_CUDA(&X, &y, &Parameters, &Train_Parameters, 150, 0.03, cost_function);
    //cutStopTimer(timer);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    //printf("\nProcessing time: %f (ms)\n", cutGetTimerValue(timer));
    printf("\nProcessing time: %f (ms)\n", milliseconds);


    return 0;
}