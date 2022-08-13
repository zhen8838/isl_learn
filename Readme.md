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
export CFLAGS=-I/Users/lisa/miniforge3/envs/dl/include # please use conda install gmp NOTE need replace by your path
export LDFLAGS=-L/Users/lisa/miniforge3/envs/dl/lib # gmp.dylib
./configure --prefix=`pwd`/build --with-clang-prefix=/Users/lisa/Documents/llvm-project/build/install # the custom llvm install path
make 

sed -i -e 's/libpet.so/libpet.dylib/g' **/*.py
sed -i -e 's/libisl.so.23/libisl.dylib/g' **/*.py
sed -i -e 's/cdll.LoadLibrary("libc.so.6")/cdll.LoadLibrary("libc.dylib")/g' **/*.py
export PYTHONPATH="`pwd`/interface:`pwd`/isl/interface:$PYTHONPATH"
export DYLD_LIBRARY_PATH="`pwd`/.libs:`pwd`/isl/.libs:/Users/lisa/Documents/llvm-project/build/install/lib:$DYLD_LIBRARY_PATH"
```

jupyter nbconvert *.ipynb --to markdown --output *.md
jupyter nbconvert 01_presburger_sets.ipynb  --to markdown --output 01_presburger_sets.md
jupyter nbconvert 02_iteration-domains.ipynb  --to markdown --output 02_iteration-domains.md
jupyter nbconvert 03_schedules.ipynb  --to markdown --output 03_schedules.md
jupyter nbconvert 04_memory.ipynb  --to markdown --output 04_memory.md
jupyter nbconvert 05_dependences.ipynb  --to markdown --output 05_dependences.md
jupyter nbconvert 06_classical-loop-transformations.ipynb  --to markdown --output 06_classical-loop-transformations.md
jupyter nbconvert 07_ast-generation.ipynb  --to markdown --output 07_ast-generation.md
jupyter nbconvert 08_c-parser.ipynb  --to markdown --output 08_c-parser.md