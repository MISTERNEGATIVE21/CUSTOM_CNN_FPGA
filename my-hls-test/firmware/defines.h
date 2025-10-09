#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 3
#define N_INPUT_2_1 128
#define N_INPUT_3_1 128
#define OUT_DEPTH_43 128
#define OUT_HEIGHT_43 128
#define OUT_WIDTH_43 3
#define OUT_DEPTH_43 128
#define OUT_HEIGHT_43 128
#define OUT_WIDTH_43 3
#define OUT_HEIGHT_92 128
#define OUT_WIDTH_92 128
#define N_FILT_92 32
#define OUT_HEIGHT_45 128
#define OUT_WIDTH_45 128
#define N_FILT_45 32
#define OUT_HEIGHT_45 128
#define OUT_WIDTH_45 128
#define N_FILT_45 32
#define OUT_HEIGHT_45 128
#define OUT_WIDTH_45 128
#define N_FILT_45 32
#define OUT_HEIGHT_45 128
#define OUT_WIDTH_45 128
#define N_FILT_45 32
#define OUT_HEIGHT_48 64
#define OUT_WIDTH_48 64
#define N_FILT_48 32
#define OUT_HEIGHT_93 64
#define OUT_WIDTH_93 64
#define N_FILT_93 64
#define OUT_HEIGHT_49 64
#define OUT_WIDTH_49 64
#define N_FILT_49 64
#define OUT_HEIGHT_49 64
#define OUT_WIDTH_49 64
#define N_FILT_49 64
#define OUT_HEIGHT_49 64
#define OUT_WIDTH_49 64
#define N_FILT_49 64
#define OUT_HEIGHT_49 64
#define OUT_WIDTH_49 64
#define N_FILT_49 64
#define OUT_HEIGHT_52 32
#define OUT_WIDTH_52 32
#define N_FILT_52 64
#define OUT_HEIGHT_94 32
#define OUT_WIDTH_94 32
#define N_FILT_94 128
#define OUT_HEIGHT_53 32
#define OUT_WIDTH_53 32
#define N_FILT_53 128
#define OUT_HEIGHT_53 32
#define OUT_WIDTH_53 32
#define N_FILT_53 128
#define OUT_HEIGHT_53 32
#define OUT_WIDTH_53 32
#define N_FILT_53 128
#define OUT_HEIGHT_53 32
#define OUT_WIDTH_53 32
#define N_FILT_53 128
#define OUT_HEIGHT_56 16
#define OUT_WIDTH_56 16
#define N_FILT_56 128
#define N_SIZE_0_57 32768
#define N_LAYER_90 256
#define N_LAYER_58 256
#define N_LAYER_58 256
#define N_LAYER_58 256
#define N_LAYER_91 13


// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<16,6> layer43_t;
typedef ap_fixed<8,8,AP_RND_CONV,AP_SAT,0> layer69_t;
typedef ap_fixed<16,6> model_default_t;
typedef ap_fixed<32,22> Conv2D_Conv_0_result_t;
typedef ap_fixed<8,8,AP_RND_CONV,AP_SAT_SYM,0> weight92_t;
typedef ap_fixed<49,29> Quant_0_rescale_result_t;
typedef ap_fixed<16,6> layer46_t;
typedef ap_fixed<18,8> Relu_0_table_t;
typedef ap_ufixed<8,8,AP_RND_CONV,AP_SAT,0> layer72_t;
typedef ap_fixed<25,15> Quant_5_rescale_result_t;
typedef ap_fixed<25,15> layer48_t;
typedef ap_fixed<43,33> Conv2D_Conv_1_result_t;
typedef ap_fixed<8,8,AP_RND_CONV,AP_SAT_SYM,0> weight93_t;
typedef ap_fixed<60,40> Quant_1_rescale_result_t;
typedef ap_fixed<16,6> layer50_t;
typedef ap_fixed<18,8> Relu_1_table_t;
typedef ap_ufixed<8,8,AP_RND_CONV,AP_SAT,0> layer75_t;
typedef ap_fixed<25,15> Quant_6_rescale_result_t;
typedef ap_fixed<25,15> layer52_t;
typedef ap_fixed<44,34> Conv2D_Conv_2_result_t;
typedef ap_fixed<8,8,AP_RND_CONV,AP_SAT_SYM,0> weight94_t;
typedef ap_fixed<61,41> Quant_2_rescale_result_t;
typedef ap_fixed<16,6> layer54_t;
typedef ap_fixed<18,8> Relu_2_table_t;
typedef ap_ufixed<8,8,AP_RND_CONV,AP_SAT,0> layer78_t;
typedef ap_fixed<25,15> Quant_7_rescale_result_t;
typedef ap_fixed<25,15> layer56_t;
typedef ap_fixed<49,39> Dense_MatMul_0_result_t;
typedef ap_fixed<8,8,AP_RND_CONV,AP_SAT_SYM,0> weight90_t;
typedef ap_uint<1> layer90_index;
typedef ap_fixed<66,46> bn_Add_0_result_t;
typedef ap_fixed<16,6> layer60_t;
typedef ap_fixed<18,8> Relu_3_table_t;
typedef ap_ufixed<8,8,AP_RND_CONV,AP_SAT,0> layer81_t;
typedef ap_fixed<33,23> result_t;
typedef ap_uint<1> layer91_index;


#endif
