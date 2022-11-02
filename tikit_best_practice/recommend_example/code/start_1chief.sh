#!/bin/bash
set -x
export CHIEF_OR_EVALUATOR=1

sh start.sh main.py "$@"
