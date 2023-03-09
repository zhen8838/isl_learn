# Lesson

1. Presburger Sets and Relations
2. Iteration Domains
3. Schedules
4. Memory Access Analysis
5. Dependence Analysis
6. Classical Loop Transformations
7. AST Generation
8. Parsing C Code

# Install Pet Python Interface On Mac M1

```sh
cd pet/
# use brew install automake
sed -i -e 's/^AM_INIT_AUTOMAKE.*/AM_INIT_AUTOMAKE/g' **/configure.ac
sed -i -e 's/s\/-L\/-R\/g/s\/-L\/-Wl,-rpath,\/g/g' **/configure # setup rpath
sed -i -e 's/s\/-L\/-R\/g/s\/-L\/-Wl,-rpath,\/g/g' **/ax_detect_clang.m4 # avoid override by reconfigure.
export CFLAGS=-I/Users/lisa/miniforge3/envs/ci/include # please use conda install gmp NOTE need replace by your path
export LDFLAGS=-L/Users/lisa/miniforge3/envs/ci/lib # gmp.dylib
./configure --prefix=`pwd`/build --with-clang-prefix=/Users/lisa/Documents/llvm-project/build/install # the custom llvm install path
make 

sed -i -e 's/libpet.so/libpet.dylib/g' **/*.py
sed -i -e 's/libisl.so.23/libisl.dylib/g' **/*.py
sed -i -e 's/cdll.LoadLibrary("libc.so.6")/cdll.LoadLibrary("libc.dylib")/g' **/*.py
export PYTHONPATH="`pwd`/interface:`pwd`/isl/interface:$PYTHONPATH"
export DYLD_LIBRARY_PATH="`pwd`/.libs:`pwd`/isl/.libs:/Users/lisa/Documents/llvm-project/build/install/lib:$DYLD_LIBRARY_PATH"
```