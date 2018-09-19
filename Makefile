
.PHONY: build run run_binarytrees_cpp run_binarytrees_java build_binarytrees_java build_binarytrees_cpp

all: build run

run: run_binarytrees_cpp run_binarytrees_java run_binarytrees_go
	
build: build_binarytrees_java build_binarytrees_cpp build_binarytrees_go

###### run #####
run_binarytrees_cpp: build_binarytrees_cpp
	cd cpp && time ./binarytrees 21

run_binarytrees_java: build_binarytrees_java
	cd java && time java binarytrees 21

run_binarytrees_go: build_binarytrees_go
	cd go && time ./binarytrees 21

###### build #####
build_binarytrees_cpp: cpp/binarytrees.cpp
	cd cpp && g++ -O3 -DNDEBUG -fomit-frame-pointer -march=native -std=c++11 -fopenmp -o binarytrees binarytrees.cpp

build_binarytrees_java: java/binarytrees.java
	cd java && javac -d .  binarytrees.java

build_binarytrees_go: go/binarytrees.go
	cd go && GOPATH=`pwd` go build binarytrees.go
