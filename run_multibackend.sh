#!/bin/bash -e
echo "********* CPU *********"
CPU=1 python3 $@
echo "********* GPU *********"
GPU=1 python3 $@
echo "********* METAL *********"
METAL=1 python3 $@
echo "********* CLANG *********"
CLANG=1 python3 $@
echo "********* LLVM *********"
LLVM=1 python3 $@
