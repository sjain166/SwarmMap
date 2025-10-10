#!/bin/bash
set -e

export DISPLAY=:0
export LD_LIBRARY_PATH=/opt/SwarmMap/lib:/opt/SwarmMap/code/Thirdparty/DBoW2/lib:/opt/SwarmMap/code/Thirdparty/g2o/lib:/usr/local/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# Wait for server to be ready
sleep 3

cd /opt/SwarmMap
exec ./bin/swarm_client -v code/Vocabulary/ORBvoc.bin -d config/client.yaml -l info --viewer 0 --mapviewer 0
