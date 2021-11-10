#!/bin/bash
for i in {1...100}
do
    pytest test_basic.py -v
done

