#!/bin/bash

echo "python3 script.py -n 5 -m node_influence/PageRank.py -p results"
python3 script.py -n 5 -m node_influence/WaveletTransform.py -p results

echo "python3 script.py -n 5 -m node_influence/StandardDeviation.py -p results"
python3 script.py -n 5 -m node_influence/StandardDeviation.py -p results

echo "python3 script.py -n 5 -m node_influence/SGT_BetweennessCentrality.py -p results"
python3 script.py -n 5 -m node_influence/SGT_BetweennessCentrality.py -p results

echo "python3 script.py -n 5 -m node_influence/SGT_ClosenessCentrality.py -p results"
python3 script.py -n 5 -m node_influence/SGT_ClosenessCentrality.py -p results

echo "python3 script.py -n 5 -m node_influence/SGT_Laplacian.py -p results"
python3 script.py -n 5 -m node_influence/SGT_Laplacian.py -p results

