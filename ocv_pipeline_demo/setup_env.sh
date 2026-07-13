#!/bin/bash
set -e

echo "============================================================"
echo "  Setting up OpenCV 5 + ROCm/HIP Environment"
echo "============================================================"

# --- Install system dependencies ---
echo "[setup] Installing ffmpeg and build dependencies..."
apt-get update -qq
apt-get install -y -qq ffmpeg libavcodec-dev libavformat-dev libavutil-dev \
    libswscale-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    pkg-config libpng-dev libjpeg-dev libtiff-dev 2>/dev/null

# --- Set up MIGraphX Python path ---
echo "[setup] Configuring environment..."
export PYTHONPATH=/opt/rocm/lib:${PYTHONPATH}
if ! grep -q "PYTHONPATH=/opt/rocm/lib" ~/.bashrc 2>/dev/null; then
    echo 'export PYTHONPATH=/opt/rocm/lib:${PYTHONPATH}' >> ~/.bashrc
fi

# --- Build OpenCV 5 + HIP ---
OPENCV_SRC=/home/opencv
OPENCV_CONTRIB=/home/opencv_contrib
OPENCV_BUILD=/home/opencv/build
OPENCV_INSTALL=/opt/opencv5

if [ -f "${OPENCV_INSTALL}/lib/python3/dist-packages/cv2/cv2.so" ] || \
   [ -f "${OPENCV_INSTALL}/lib/python3.12/dist-packages/cv2/cv2.cpython-312-x86_64-linux-gnu.so" ]; then
    echo "[setup] OpenCV 5 + HIP already installed at ${OPENCV_INSTALL}"
else
    echo "[setup] Building OpenCV 5 + HIP (this will take ~15-30 minutes)..."

    # Ensure we're on the 5.x-hip branch
    cd ${OPENCV_SRC}
    CURRENT_BRANCH=$(git branch --show-current)
    if [ "${CURRENT_BRANCH}" != "5.x-hip" ]; then
        echo "[setup] Switching to 5.x-hip branch..."
        git checkout 5.x-hip
    fi

    mkdir -p ${OPENCV_BUILD}
    cd ${OPENCV_BUILD}

    echo "[setup] Running cmake..."
    cmake -DWITH_HIP=ON \
          -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
          -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/amdclang++ \
          -DCMAKE_PREFIX_PATH=/opt/rocm \
          -DOPENCV_EXTRA_MODULES_PATH=${OPENCV_CONTRIB}/modules \
          -DWITH_CUDA=OFF \
          -DWITH_OPENCL=ON \
          -DWITH_FFMPEG=ON \
          -DBUILD_opencv_python3=ON \
          -DBUILD_TESTS=OFF \
          -DBUILD_PERF_TESTS=OFF \
          -DBUILD_EXAMPLES=OFF \
          -DBUILD_DOCS=OFF \
          -DCMAKE_INSTALL_PREFIX=${OPENCV_INSTALL} \
          -DCMAKE_BUILD_TYPE=Release \
          ${OPENCV_SRC} 2>&1 | tee cmake_output.log

    echo "[setup] Building (using $(nproc) cores)..."
    cmake --build . -j$(nproc) 2>&1 | tee build_output.log

    echo "[setup] Installing..."
    cmake --install .

    echo "[setup] Configuring Python path..."
    # Find where cv2 was installed and add to PYTHONPATH
    CV2_PATH=$(find ${OPENCV_INSTALL} -name "cv2*.so" -path "*/python*" | head -1)
    if [ -n "${CV2_PATH}" ]; then
        CV2_DIR=$(dirname $(dirname ${CV2_PATH}))
        echo "export PYTHONPATH=${CV2_DIR}:\${PYTHONPATH}" >> ~/.bashrc
        export PYTHONPATH=${CV2_DIR}:${PYTHONPATH}
        echo "[setup] Added ${CV2_DIR} to PYTHONPATH"
    fi
fi

# --- Verify ---
echo ""
echo "[setup] Verification:"
echo -n "  ffmpeg: "; ffmpeg -version 2>&1 | head -1
echo -n "  OpenCV: "; python3 -c "import cv2; print(cv2.__version__)" 2>&1
echo -n "  HIP GPU devices: "; python3 -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())" 2>&1
echo -n "  MIGraphX: "; PYTHONPATH=/opt/rocm/lib:${PYTHONPATH} python3 -c "import migraphx; print('OK')" 2>&1
echo -n "  vLLM: "; python3 -c "import vllm; print(vllm.__version__)" 2>&1

echo ""
echo "============================================================"
echo "  Setup complete!"
echo "============================================================"
