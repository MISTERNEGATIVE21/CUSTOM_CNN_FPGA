#include <iostream>

#include "myproject.h"
#include "parameters.h"


void myproject(
    input_t global_in[N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1],
    result_t layer91_out[N_LAYER_91]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=global_in complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer91_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=global_in,layer91_out 
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<model_default_t, 49152>(s69, "s69.txt");
        nnet::load_weights_from_txt<model_default_t, 49152>(b69, "b69.txt");
        nnet::load_weights_from_txt<weight92_t, 864>(w92, "w92.txt");
        nnet::load_weights_from_txt<model_default_t, 32>(b92, "b92.txt");
        nnet::load_weights_from_txt<model_default_t, 524288>(s85, "s85.txt");
        nnet::load_weights_from_txt<model_default_t, 524288>(b85, "b85.txt");
        nnet::load_weights_from_txt<model_default_t, 524288>(s72, "s72.txt");
        nnet::load_weights_from_txt<model_default_t, 524288>(b72, "b72.txt");
        nnet::load_weights_from_txt<model_default_t, 524288>(s73, "s73.txt");
        nnet::load_weights_from_txt<model_default_t, 524288>(b73, "b73.txt");
        nnet::load_weights_from_txt<weight93_t, 18432>(w93, "w93.txt");
        nnet::load_weights_from_txt<model_default_t, 64>(b93, "b93.txt");
        nnet::load_weights_from_txt<model_default_t, 262144>(s86, "s86.txt");
        nnet::load_weights_from_txt<model_default_t, 262144>(b86, "b86.txt");
        nnet::load_weights_from_txt<model_default_t, 262144>(s75, "s75.txt");
        nnet::load_weights_from_txt<model_default_t, 262144>(b75, "b75.txt");
        nnet::load_weights_from_txt<model_default_t, 262144>(s76, "s76.txt");
        nnet::load_weights_from_txt<model_default_t, 262144>(b76, "b76.txt");
        nnet::load_weights_from_txt<weight94_t, 73728>(w94, "w94.txt");
        nnet::load_weights_from_txt<model_default_t, 128>(b94, "b94.txt");
        nnet::load_weights_from_txt<model_default_t, 131072>(s87, "s87.txt");
        nnet::load_weights_from_txt<model_default_t, 131072>(b87, "b87.txt");
        nnet::load_weights_from_txt<model_default_t, 131072>(s78, "s78.txt");
        nnet::load_weights_from_txt<model_default_t, 131072>(b78, "b78.txt");
        nnet::load_weights_from_txt<model_default_t, 131072>(s79, "s79.txt");
        nnet::load_weights_from_txt<model_default_t, 131072>(b79, "b79.txt");
        nnet::load_weights_from_txt<weight90_t, 8388608>(w90, "w90.txt");
        nnet::load_weights_from_txt<model_default_t, 256>(b90, "b90.txt");
        nnet::load_weights_from_txt<model_default_t, 256>(s88, "s88.txt");
        nnet::load_weights_from_txt<model_default_t, 256>(b88, "b88.txt");
        nnet::load_weights_from_txt<model_default_t, 256>(s81, "s81.txt");
        nnet::load_weights_from_txt<model_default_t, 256>(b81, "b81.txt");
        nnet::load_weights_from_txt<model_default_t, 3328>(w91, "w91.txt");
        nnet::load_weights_from_txt<model_default_t, 13>(b91, "b91.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    layer43_t layer43_out[OUT_DEPTH_43*OUT_HEIGHT_43*OUT_WIDTH_43];
    #pragma HLS ARRAY_PARTITION variable=layer43_out complete dim=0
    nnet::transpose<input_t, layer43_t, config43>(global_in, layer43_out); // Transpose_0

    layer69_t layer69_out[OUT_DEPTH_43*OUT_HEIGHT_43*OUT_WIDTH_43];
    #pragma HLS ARRAY_PARTITION variable=layer69_out complete dim=0
    nnet::normalize<layer43_t, layer69_t, config69>(layer43_out, layer69_out, s69, b69); // Quant_4_scale

    Conv2D_Conv_0_result_t layer92_out[OUT_HEIGHT_92*OUT_WIDTH_92*N_FILT_92];
    #pragma HLS ARRAY_PARTITION variable=layer92_out complete dim=0
    nnet::conv_2d_cl<layer69_t, Conv2D_Conv_0_result_t, config92>(layer69_out, layer92_out, w92, b92); // Conv2D_Conv_0

    Quant_0_rescale_result_t layer85_out[OUT_HEIGHT_45*OUT_WIDTH_45*N_FILT_45];
    #pragma HLS ARRAY_PARTITION variable=layer85_out complete dim=0
    nnet::normalize<Conv2D_Conv_0_result_t, Quant_0_rescale_result_t, config85>(layer92_out, layer85_out, s85, b85); // Quant_0_rescale

    layer46_t layer46_out[OUT_HEIGHT_45*OUT_WIDTH_45*N_FILT_45];
    #pragma HLS ARRAY_PARTITION variable=layer46_out complete dim=0
    nnet::relu<Quant_0_rescale_result_t, layer46_t, ReLU_config46>(layer85_out, layer46_out); // Relu_0

    layer72_t layer72_out[OUT_HEIGHT_45*OUT_WIDTH_45*N_FILT_45];
    #pragma HLS ARRAY_PARTITION variable=layer72_out complete dim=0
    nnet::normalize<layer46_t, layer72_t, config72>(layer46_out, layer72_out, s72, b72); // Quant_5_scale

    Quant_5_rescale_result_t layer73_out[OUT_HEIGHT_45*OUT_WIDTH_45*N_FILT_45];
    #pragma HLS ARRAY_PARTITION variable=layer73_out complete dim=0
    nnet::normalize<layer72_t, Quant_5_rescale_result_t, config73>(layer72_out, layer73_out, s73, b73); // Quant_5_rescale

    layer48_t layer48_out[OUT_HEIGHT_48*OUT_WIDTH_48*N_FILT_48];
    #pragma HLS ARRAY_PARTITION variable=layer48_out complete dim=0
    nnet::pooling2d_cl<Quant_5_rescale_result_t, layer48_t, config48>(layer73_out, layer48_out); // MaxPool_0

    Conv2D_Conv_1_result_t layer93_out[OUT_HEIGHT_93*OUT_WIDTH_93*N_FILT_93];
    #pragma HLS ARRAY_PARTITION variable=layer93_out complete dim=0
    nnet::conv_2d_cl<layer48_t, Conv2D_Conv_1_result_t, config93>(layer48_out, layer93_out, w93, b93); // Conv2D_Conv_1

    Quant_1_rescale_result_t layer86_out[OUT_HEIGHT_49*OUT_WIDTH_49*N_FILT_49];
    #pragma HLS ARRAY_PARTITION variable=layer86_out complete dim=0
    nnet::normalize<Conv2D_Conv_1_result_t, Quant_1_rescale_result_t, config86>(layer93_out, layer86_out, s86, b86); // Quant_1_rescale

    layer50_t layer50_out[OUT_HEIGHT_49*OUT_WIDTH_49*N_FILT_49];
    #pragma HLS ARRAY_PARTITION variable=layer50_out complete dim=0
    nnet::relu<Quant_1_rescale_result_t, layer50_t, ReLU_config50>(layer86_out, layer50_out); // Relu_1

    layer75_t layer75_out[OUT_HEIGHT_49*OUT_WIDTH_49*N_FILT_49];
    #pragma HLS ARRAY_PARTITION variable=layer75_out complete dim=0
    nnet::normalize<layer50_t, layer75_t, config75>(layer50_out, layer75_out, s75, b75); // Quant_6_scale

    Quant_6_rescale_result_t layer76_out[OUT_HEIGHT_49*OUT_WIDTH_49*N_FILT_49];
    #pragma HLS ARRAY_PARTITION variable=layer76_out complete dim=0
    nnet::normalize<layer75_t, Quant_6_rescale_result_t, config76>(layer75_out, layer76_out, s76, b76); // Quant_6_rescale

    layer52_t layer52_out[OUT_HEIGHT_52*OUT_WIDTH_52*N_FILT_52];
    #pragma HLS ARRAY_PARTITION variable=layer52_out complete dim=0
    nnet::pooling2d_cl<Quant_6_rescale_result_t, layer52_t, config52>(layer76_out, layer52_out); // MaxPool_1

    Conv2D_Conv_2_result_t layer94_out[OUT_HEIGHT_94*OUT_WIDTH_94*N_FILT_94];
    #pragma HLS ARRAY_PARTITION variable=layer94_out complete dim=0
    nnet::conv_2d_cl<layer52_t, Conv2D_Conv_2_result_t, config94>(layer52_out, layer94_out, w94, b94); // Conv2D_Conv_2

    Quant_2_rescale_result_t layer87_out[OUT_HEIGHT_53*OUT_WIDTH_53*N_FILT_53];
    #pragma HLS ARRAY_PARTITION variable=layer87_out complete dim=0
    nnet::normalize<Conv2D_Conv_2_result_t, Quant_2_rescale_result_t, config87>(layer94_out, layer87_out, s87, b87); // Quant_2_rescale

    layer54_t layer54_out[OUT_HEIGHT_53*OUT_WIDTH_53*N_FILT_53];
    #pragma HLS ARRAY_PARTITION variable=layer54_out complete dim=0
    nnet::relu<Quant_2_rescale_result_t, layer54_t, ReLU_config54>(layer87_out, layer54_out); // Relu_2

    layer78_t layer78_out[OUT_HEIGHT_53*OUT_WIDTH_53*N_FILT_53];
    #pragma HLS ARRAY_PARTITION variable=layer78_out complete dim=0
    nnet::normalize<layer54_t, layer78_t, config78>(layer54_out, layer78_out, s78, b78); // Quant_7_scale

    Quant_7_rescale_result_t layer79_out[OUT_HEIGHT_53*OUT_WIDTH_53*N_FILT_53];
    #pragma HLS ARRAY_PARTITION variable=layer79_out complete dim=0
    nnet::normalize<layer78_t, Quant_7_rescale_result_t, config79>(layer78_out, layer79_out, s79, b79); // Quant_7_rescale

    layer56_t layer56_out[OUT_HEIGHT_56*OUT_WIDTH_56*N_FILT_56];
    #pragma HLS ARRAY_PARTITION variable=layer56_out complete dim=0
    nnet::pooling2d_cl<Quant_7_rescale_result_t, layer56_t, config56>(layer79_out, layer56_out); // MaxPool_2

    auto& layer57_out = layer56_out;
    Dense_MatMul_0_result_t layer90_out[N_LAYER_90];
    #pragma HLS ARRAY_PARTITION variable=layer90_out complete dim=0
    nnet::dense<layer56_t, Dense_MatMul_0_result_t, config90>(layer57_out, layer90_out, w90, b90); // Dense_MatMul_0

    bn_Add_0_result_t layer88_out[N_LAYER_58];
    #pragma HLS ARRAY_PARTITION variable=layer88_out complete dim=0
    nnet::normalize<Dense_MatMul_0_result_t, bn_Add_0_result_t, config88>(layer90_out, layer88_out, s88, b88); // bn_Add_0

    layer60_t layer60_out[N_LAYER_58];
    #pragma HLS ARRAY_PARTITION variable=layer60_out complete dim=0
    nnet::relu<bn_Add_0_result_t, layer60_t, ReLU_config60>(layer88_out, layer60_out); // Relu_3

    layer81_t layer81_out[N_LAYER_58];
    #pragma HLS ARRAY_PARTITION variable=layer81_out complete dim=0
    nnet::normalize<layer60_t, layer81_t, config81>(layer60_out, layer81_out, s81, b81); // Quant_8_scale

    nnet::dense<layer81_t, result_t, config91>(layer81_out, layer91_out, w91, b91); // Dense_MatMul_1

}

