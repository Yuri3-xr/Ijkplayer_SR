LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)
LOCAL_MODULE := tensorflowlite
LOCAL_SRC_FILES := ../native_libs/$(TARGET_ARCH_ABI)/libtensorflowlite.so
include $(PREBUILT_SHARED_LIBRARY)
include $(CLEAR_VARS)
LOCAL_MODULE := tensorflowlite_gpu
LOCAL_SRC_FILES := ../native_libs/$(TARGET_ARCH_ABI)/libtensorflowlite_gpu_delegate.so
include $(PREBUILT_SHARED_LIBRARY)
include $(CLEAR_VARS)
LOCAL_MODULE := xjtu_sr
LOCAL_SRC_FILES := xjtu_sr.cpp
LOCAL_C_INCLUDES += $(LOCAL_PATH)/../../extra/ffmpeg
LOCAL_C_INCLUDES += $(LOCAL_PATH)/../include
LOCAL_SHARED_LIBRARIES += tensorflowlite
LOCAL_SHARED_LIBRARIES += tensorflowlite_gpu
LOCAL_CPPFLAGS += "-std=c++17"
include $(BUILD_SHARED_LIBRARY)
