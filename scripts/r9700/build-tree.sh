#!/bin/bash
# build-tree.sh <tree> <build-dir-name> [targets...]: static Vulkan build (clang, NATIVE), same flags for every tree
T=$1; D=$2; shift; shift
TARGETS=${*:-llama-server llama-cli llama-completion llama-bench llama-perplexity llama-eval-callback test-backend-ops}
cd "$T" || exit 1
[ -f $D/CMakeCache.txt ] || cmake -S . -B $D -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DBUILD_SHARED_LIBS=OFF -DCMAKE_EXE_LINKER_FLAGS="-static-libstdc++ -static-libgcc" -DGGML_VULKAN=ON -DGGML_NATIVE=ON -DLLAMA_OPENSSL=OFF \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_MTMD=ON -DLLAMA_BUILD_UI=OFF -DVulkan_GLSLC_EXECUTABLE=$HOME/src/shaderc/build/glslc/glslc > $D-cfg.log 2>&1 \
  || { echo CFG-FAIL; tail -20 $D-cfg.log; exit 1; }
( time cmake --build $D -j 32 --target $TARGETS ) > $D.log 2>&1
rc=$?; echo "build $T/$D rc=$rc ($(grep real $D.log))"; grep -m8 'error:' $D.log
exit $rc
