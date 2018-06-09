#include <cudnn.h>
#include <cassert>
#include <cstdlib>
#include <iostream>

int convolution(){
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    //input 
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                        /*format=*/CUDNN_TENSOR_NHWC,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/1,
                                        /*channels=*/3,
                                        /*image_height=*/image.rows,
                                        /*image_width=*/image.cols));

    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                        /*format=*/CUDNN_TENSOR_NHWC,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/1,
                                        /*channels=*/3,
                                        /*image_height=*/image.rows,
                                        /*image_width=*/image.cols));

    cudnnFilterDescriptor_t kernel_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                            /*dataType=*/CUDNN_DATA_FLOAT,
                                            /*format=*/CUDNN_TENSOR_NCHW,
                                            /*out_channels=*/3,
                                            /*in_channels=*/3,
                                            /*kernel_height=*/3,
                                            /*kernel_width=*/3));

    cudnnConvolutionDescriptor_t convolution_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                                /*pad_height=*/1,
                                                /*pad_width=*/1,
                                                /*vertical_stride=*/1,
                                                /*horizontal_stride=*/1,
                                                /*dilation_height=*/1,
                                                /*dilation_width=*/1,
                                                /*mode=*/CUDNN_CROSS_CORRELATION,
                                                /*computeType=*/CUDNN_DATA_FLOAT));

    cudnnConvolutionFwdAlgo_t convolution_algorithm;
    checkCUDNN(
        cudnnGetConvolutionForwardAlgorithm(cudnn,
                                            input_descriptor,
                                            kernel_descriptor,
                                            convolution_descriptor,
                                            output_descriptor,
                                            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                            /*memoryLimitInBytes=*/0,
                                            &convolution_algorithm));

    size_t workspace_bytes = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                        input_descriptor,
                                                        kernel_descriptor,
                                                        convolution_descriptor,
                                                        output_descriptor,
                                                        convolution_algorithm,
                                                        &workspace_bytes));
    std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB"
                << std::endl;
    float* h_output = new float[image_bytes];
    cudaMemcpy(h_output, d_output, image_bytes, cudaMemcpyDeviceToHost);
    
    // Do something with h_output ...
    
    delete[] h_output;
    cudaFree(d_kernel);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_workspace);
    
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
    
    cudnnDestroy(cudnn);
}