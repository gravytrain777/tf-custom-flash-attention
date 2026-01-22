export LD_LIBRARY_PATH=/usr/local/cuda-12.3/:/usr/local/isl/lib/:$LD_LIBRARY_PATH

cd ./sdpa

OP_NAME="sdpa"
sosuffix="so.2.16"

soname="${OP_NAME}.${sosuffix}"

OP_SOURCE="${OP_NAME}_ops.cc"
OP_OUT="${OP_NAME}_ops.o"

CPU_SOURCE="${OP_NAME}_cpu.cc"
CPU_OUT="${OP_NAME}_cpu.o"

GPU_FWD_SOURCE="${OP_NAME}_fwd.cu.cc"
GPU_FWD_OUT="${OP_NAME}_fwd.cu.o"

GPU_BWD_SOURCE="${OP_NAME}_bwd.cu.cc"
GPU_BWD_OUT="${OP_NAME}_bwd.cu.o"


cudapath="/usr/local/cuda-12.3"
tfpath="/your/python/env/path/site-packages/tensorflow"

TF_CFLAGS="-I$tfpath/include -D_GLIBCXX_USE_CXX11_ABI=1 --std=c++17 -DEIGEN_MAX_ALIGN_BYTES=64"
TF_LFLAGS="-L$tfpath -l:libtensorflow_framework.so.2"

cuda_lib_path="$cudapath/lib64"
cudart_lib="cudart"
# CUDA_LINK="-Wl,-rpath,${cuda_lib_path} -L${cuda_lib_path} -l${cudart_lib}"
CUDA_LINK="-L${cuda_lib_path} -l${cudart_lib}"

$cudapath/bin/nvcc -c ${GPU_FWD_SOURCE} -o ${GPU_FWD_OUT} \
    $TF_CFLAGS -D GOOGLE_CUDA=1 -x cu \
    -ccbin=/usr/bin/g++ -Xcompiler "-fPIC -Wno-deprecated-declarations -Wno-attributes -O2"\
    --expt-relaxed-constexpr -diag-suppress 2810,611 -O2 

$cudapath/bin/nvcc -c ${GPU_BWD_SOURCE} -o ${GPU_BWD_OUT} \
    $TF_CFLAGS -D GOOGLE_CUDA=1 -x cu \
    -ccbin=/usr/bin/g++ -Xcompiler "-fPIC -Wno-deprecated-declarations -Wno-attributes -O2"\
    --expt-relaxed-constexpr -diag-suppress 2810,611 -O2 

/usr/bin/g++ -c $OP_SOURCE -o $OP_OUT -fPIC $TF_CFLAGS -D GOOGLE_CUDA=1 -O2

/usr/bin/g++ -c $CPU_SOURCE -o $CPU_OUT -fPIC $TF_CFLAGS -D GOOGLE_CUDA=1 -O2

/usr/bin/g++ -std=c++17 -shared -o $soname $OP_OUT $CPU_OUT $GPU_FWD_OUT $GPU_BWD_OUT\
    -fPIC ${TF_LFLAGS} ${CUDA_LINK} -mavx2 -O2 \
    -march=native -mtune=native \
     -D GOOGLE_CUDA=1 
