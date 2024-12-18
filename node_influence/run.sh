#!/bin/bash


echo "@@@@@ RUNNING CommunityDetectionCentrality @@@@@"
python3 script.py -n 5 -m node_influence/CommunityDetectionCentrality.py -p results

echo "@@@@@@ RUNNING PageRank @@@@@@"
python3 script.py -n 5 -m node_influence/PageRank.py -p results

echo "@@@@@@ RUNNING WaveletTransform @@@@@@"
python3 script.py -n 5 -m node_influence/WaveletTransform.py -p results

echo "@@@@@@ RUNNING StandardDeviation @@@@@@"
python3 script.py -n 5 -m node_influence/StandardDeviation.py -p results

echo "@@@@@@ RUNNING SGT_BetweennessCentrality @@@@@@"
python3 script.py -n 5 -m node_influence/SGT_BetweennessCentrality.py -p results

echo "@@@@@@ RUNNING SGT_ClosenessCentrality @@@@@@"
python3 script.py -n 5 -m node_influence/SGT_ClosenessCentrality.py -p results

echo "@@@@@@ RUNNING SGT_Laplacian @@@@@@"
python3 script.py -n 5 -m node_influence/SGT_Laplacian.py -p results

