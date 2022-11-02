#!/bin/bash
pkill -f python
sleep 10
sh start_1evaluator.sh "$@" > 1evaluator.log 2>&1 &
sleep 1
sh start_1chief.sh "$@"