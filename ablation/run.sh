#!/bin/bash


echo "@@@@ RUNNING: NIEN.py @@@@"
python3 script_full.py -n 5 -m NIEN.py -p results

echo "@@@@ RUNNING: NIEN_FixedEncoding.py @@@@"
python3 script_full.py -n 5 -m NIEN_FixedEncoding.py -p results

echo "@@@@ RUNNING: NIEN_SingleScale.py @@@@"
python3 script_full.py -n 5 -m NIEN_SingleScale.py -p results

echo "@@@@ RUNNING: NIEN_WithoutCausalMask.py @@@@"
python3 script_full.py -n 5 -m NIEN_WithoutCausalMask.py -p results

echo "@@@@ RUNNING: NIEN_WithoutNodeInfluence.py @@@@"
python3 script_full.py -n 5 -m NIEN_WithoutNodeInfluence.py -p results
