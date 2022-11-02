#!/bin/bash
set -x
export CHIEF_OR_EVALUATOR=2

sh start.sh main.py "$@"
