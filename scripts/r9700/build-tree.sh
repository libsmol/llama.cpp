#!/bin/bash
# build-tree.sh <tree> <build-dir> [targets...]: static Vulkan build (clang, GGML_NATIVE), the same flags for every tree
# GLSLC: the glslc to use (default: the shaderc build in ~/src/shaderc, which has the coopmat / integer dot extensions)
T=$1; D=$2; shift 2
TARGETS=${*:-llama-server llama-cli llama-completion llama-bench llama-perplexity test-backend-ops}
GLSLC=${GLSLC:-$HOME/src/shaderc/build/glslc/glslc}
cd "$T" || exit 1
[ -f "$D/CMakeCache.txt" ] || cmake -S . -B "$D" -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DBUILD_SHARED_LIBS=OFF -DCMAKE_EXE_LINKER_FLAGS="-static-libstdc++ -static-libgcc" -DGGML_VULKAN=ON -DGGML_NATIVE=ON -DLLAMA_OPENSSL=OFF \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_MTMD=ON -DLLAMA_BUILD_UI=OFF -DVulkan_GLSLC_EXECUTABLE="$GLSLC" > "$D-cfg.log" 2>&1 \
  || { echo CFG-FAIL; tail -20 "$D-cfg.log"; exit 1; }
( time cmake --build "$D" -j "$(nproc)" --target $TARGETS ) > "$D.log" 2>&1
rc=$?; echo "build $T/$D rc=$rc ($(grep real "$D.log"))"; grep -m8 'error:' "$D.log"
exit $rc
