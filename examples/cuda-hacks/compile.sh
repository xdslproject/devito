#!/bin/bash

exe() { echo "$@" ; "$@" ; }

exe nvc++ -g -fPIC -std=c++11 -acc -mp -fast -shared 43471cbb098fb5ef14e2b143f162954cabb3e964.cpp -lm -o 43471cbb098fb5ef14e2b143f162954cabb3e964.so
