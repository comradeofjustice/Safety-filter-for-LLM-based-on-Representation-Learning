#!/bin/bash
cd /root/CSY/Safety-filter-for-LLM-based-on-Representation-Learning/8B
python svm_experiment.py --optimizer grid 2>&1 | tee log/experiment_stdout.log
