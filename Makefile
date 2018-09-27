
.PHONY: build run run_binarytrees_cpp run_binarytrees_java build_binarytrees_java build_binarytrees_cpp build_binarytrees_rust run_binarytrees_rust

all: run

run: run_binarytrees_cpp run_binarytrees_java run_binarytrees_rust run_binarytrees_go
	
build: build_binarytrees_java build_binarytrees_cpp build_binarytrees_go build_binarytrees_rust

CXX ?= /usr/local/builds/gcc/latest/bin/g++
#CXX = g++

###### run #####
run_binarytrees_cpp: build_binarytrees_cpp
	cd cpp && time LD_LIBRARY_PATH=`dirname $(CXX)`/../lib64 ./binarytrees 21
#	cd cpp && time ./binarytrees 21

run_binarytrees_java: build_binarytrees_java
	cd java && time java binarytrees 21

run_binarytrees_go: build_binarytrees_go
	cd go && time ./binarytrees 21

run_binarytrees_rust: build_binarytrees_rust
	cd rust && time cargo run --release --bin binarytrees 21

###### build #####
build_binarytrees_cpp: cpp/binarytrees.cpp
	cd cpp && $(CXX) -O3 -DNDEBUG -fomit-frame-pointer -march=native -std=c++17 -fopenmp -o binarytrees binarytrees.cpp

build_binarytrees_java: java/binarytrees.java
	cd java && javac -d .  binarytrees.java

build_binarytrees_go: go/binarytrees.go
	cd go && GOPATH=`pwd` go build binarytrees.go

build_binarytrees_rust: rust/src/bin/binarytrees.rs
	cd rust && cargo build --release --bin binarytrees

