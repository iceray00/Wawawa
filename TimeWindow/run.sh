#!/bin/bash


echo "@@@@ RUNNING: NIEN_12.py @@@@"
python3 script.py -n 5 -m NIEN_12.py -p results_TimeWindow

echo "@@@@ RUNNING: NIEN_6.py @@@@"
python3 script.py -n 5 -m NIEN_6.py -p results_TimeWindow

echo "@@@@ RUNNING: NIEN_3.py @@@@"
python3 script.py -n 5 -m NIEN_3.py -p results_TimeWindow

