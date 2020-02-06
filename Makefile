
.PHONY: build run run_binarytrees_cpp run_binarytrees_java build_binarytrees_java build_binarytrees_cpp build_binarytrees_rust run_binarytrees_rust

all: run

run: run_binarytrees_cpp run_binarytrees_java run_binarytrees_rust run_binarytrees_go 
	
build: build_binarytrees_java build_binarytrees_cpp build_binarytrees_go build_binarytrees_rust

CXX ?= /usr/local/builds/gcc/latest/bin/g++
#CXX ?= g++

# obtained by $(apr-1-config --cflags --cppflags --includes --link-ld)
#APRFLAGS = -pthread  -DLINUX -D_REENTRANT -D_GNU_SOURCE -I/usr/include/apr-1.0  -L/usr/lib/x86_64-linux-gnu -lapr-1
APRFLAGS = -pthread -DLINUX -D_REENTRANT -D_GNU_SOURCE -I/mnt/ldata/program/apr-1.7/include/apr-1  -L/mnt/ldata/program/apr-1.7/lib -lapr-1
NDEPTH=21
###### run #####
run_binarytrees_cpp: build_binarytrees_cpp
#	cd cpp && time ./binarytrees $(NDEPTH)
	cd cpp && time ./binarytrees.gpp-9.gpp_run $(NDEPTH)
#	cd cpp && time LD_LIBRARY_PATH=`which g++ | xargs readlink -e | xargs dirname`/../lib64 ./binarytrees 21

run_binarytrees_java: build_binarytrees_java
	cd java && java -version && time java binarytrees $(NDEPTH)

run_binarytrees_go: build_binarytrees_go
	cd go && go version && time ./binarytrees $(NDEPTH)

run_binarytrees_rust: build_binarytrees_rust
	cd rust && rustc --version && time cargo run --release --bin binarytrees $(NDEPTH)

###### build #####
build_binarytrees_cpp: cpp/binarytrees.cpp
#	cd cpp && $(CXX) -O3 -DNDEBUG -fomit-frame-pointer -march=native -std=c++17 -fopenmp -o binarytrees binarytrees.cpp
	cd cpp && $(CXX) -O3 -DNDEBUG -c -pipe -O3 -fomit-frame-pointer -march=core2  -fopenmp -I/usr/include/apr-1.0 binarytrees.gpp-9.c++ -o binarytrees.gpp-9.c++.o && $(CXX) binarytrees.gpp-9.c++.o -o binarytrees.gpp-9.gpp_run -fopenmp -lapr-1 

build_binarytrees_java: java/binarytrees.java
	cd java && javac -d .  binarytrees.java

build_binarytrees_go: go/binarytrees.go
	cd go && GOPATH=`pwd` go build binarytrees.go

build_binarytrees_rust: rust/src/bin/binarytrees.rs
	cd rust && cargo build --release --bin binarytrees

