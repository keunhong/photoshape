NUMPY=`python -c 'import numpy; print(numpy.get_include())'`
PYROOT=`python -c 'import sys; print(sys.prefix)'`
PYTHON_LIB=`python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())"`
CC=clang++
LIBS= 
#FLAGS= -Wall -DNUMPYCHECK -fPIC
#FLAGS = -Wall -DNDEBUG -O2 -ffast-math -pipe -msse -msse2 -mmmx -mfpmath=sse -fomit-frame-pointer 
#FLAGS = -Wall -DNDEBUG -O2 -ffast-math -fPIC
FLAGS = -DNUMPYCHECK -DNDEBUG -O2 -ffast-math -msse2 -fPIC

.PHONY: all
all: features_pedro_py.so

features_pedro_py.so: features_pedro_py.o
	$(CC) $^ -shared -o $@ $(LIBS)

features_pedro_py.o: features_pedro_py.cc numpymacros.h
	$(CC) -c $< $(FLAGS) -I$(NUMPY) -I$(PYTHON_LIB) -I../src/ -o $@

.PHONY: clean
clean:
	rm -f *.o *.so
