#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/string_type.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/delegates/gpu/gl_delegate.h"
#include "tensorflow/lite/stderr_reporter.h"
#include "tensorflow/lite/delegates/serialization.h"
#include "stdlib.h"
#include "xjtu_sr.h"
#include <vector>

static int xjtu_model_load = 0;
static std::unique_ptr<tflite::FlatBufferModel> model;
static tflite::ops::builtin::BuiltinOpResolver resolver;
static std::unique_ptr<tflite::Interpreter> interpreter;
static TfLiteDelegate* delegate;
static TfLiteGpuDelegateOptionsV2 options;

extern "C" void xjtu_get_sr_result(AVFrame *frame, int bigendian_switch) {
	//int xjtu_model_load = 0;
	//std::unique_ptr<tflite::FlatBufferModel> model;
	//tflite::ops::builtin::BuiltinOpResolver resolver;
	//std::unique_ptr<tflite::Interpreter> interpreter;
	//TfLiteDelegate* delegate;
	//TfLiteGpuDelegateOptionsV2 options;
	
	// 1. 导入模型，映射到内存
    if (xjtu_model_load == 0) {
		model=tflite::FlatBufferModel::BuildFromFile("/sdcard/model.tflite");
		xjtu_model_load = 1;
		// 2. 构建Tensorflow模型的解释器 interpreter
		tflite::InterpreterBuilder(*model, resolver)(&interpreter);
		// Create GPU delegate
		options = TfLiteGpuDelegateOptionsV2Default();
		options.is_precision_loss_allowed = 1;
		options.experimental_flags = TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_QUANT;
		options.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
		options.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
		options.inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
		options.inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
		delegate = TfLiteGpuDelegateV2Create(&options);
		interpreter->ModifyGraphWithDelegate(delegate);
		// Allow FP16 Inference
		interpreter->SetAllowFp16PrecisionForFp32(true);
		// Resize Input Tensor for Frames
		// So, this function just for one video.
		// In demo, just be simple.
		int h = frame->height;
		int w = frame->width;
		std::vector<int> shape;
		shape.push_back(1);
		shape.push_back(h);
		shape.push_back(w);
		shape.push_back(3);
		interpreter->ResizeInputTensor(0, shape);
		interpreter->AllocateTensors();
	}
    // 3. 构建输入Tensor
    float* input = interpreter->typed_input_tensor<float>(0);
    for (int row = 0; row < frame->height; row++) {
		for (int i = 0; i < frame->width; i++) {
			// 0BGR or RGB0 data is packed
			int idx_t = (row * frame->width + i) * 3;
			int idx_f = row * frame->linesize[0] + i * 4;
			if (bigendian_switch == 1) {
				// 0BGR
				input[idx_t + 0] = frame->data[0][idx_f + 3];
				input[idx_t + 1] = frame->data[0][idx_f + 2];
				input[idx_t + 2] = frame->data[0][idx_f + 1];
			}
			else {
				// RGB0
				input[idx_t + 0] = frame->data[0][idx_f + 0];
				input[idx_t + 1] = frame->data[0][idx_f + 1];
				input[idx_t + 2] = frame->data[0][idx_f + 2];
			}
		}
	}
    // 4. 运行模型，推导结果
    interpreter->Invoke();
    // 5. 获取输出结果
    float* output=interpreter->typed_output_tensor<float>(0);
	// 6. 写回结果
	frame->width *= 4;
	frame->height *= 4;
	frame->linesize[0] = ((32 - ((frame->width * 4) % 32)) % 32) + frame->width * 4;
	free(frame->data[0]);
	frame->data[0] = (uint8_t *) malloc(frame->height * frame->linesize[0]);
    for (int row = 0; row < frame->height; row++) {
		for (int i = 0; i < frame->width; i++) {
			// Data is packed
			int idx_t = (row * frame->width + i) * 3;
			int idx_f = row * frame->linesize[0] + i * 4;
			if (bigendian_switch == 1) {
				// 0BGR
				frame->data[0][idx_f + 1] = (uint8_t) output[idx_t + 2];
				frame->data[0][idx_f + 2] = (uint8_t) output[idx_t + 1];
				frame->data[0][idx_f + 3] = (uint8_t) output[idx_t + 0];
			}
			else {
				// RGB0
				frame->data[0][idx_f + 0] = (uint8_t) output[idx_t + 0];
				frame->data[0][idx_f + 1] = (uint8_t) output[idx_t + 1];
				frame->data[0][idx_f + 2] = (uint8_t) output[idx_t + 2];
			}
		}
	}
}
