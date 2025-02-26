#!/bin/bash
cp -rf build.lite.linux.loongarch.gcc/inference_lite_lib/cxx/lib/libpaddle_full_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/loongarch64/lib/
cp -rf build.lite.linux.loongarch.gcc/inference_lite_lib/cxx/lib/libpaddle_light_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/loongarch64/lib/
cp -rf build.lite.linux.loongarch.gcc/inference_lite_lib/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/linux/loongarch64/include/
