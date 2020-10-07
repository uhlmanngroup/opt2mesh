#! /usr/bin/env bash

# Aggregation of results for LSF mail using datamash for:
#   ⋅ CPU time
#   ⋅ Run time
#   ⋅ Turnaround time
#   ⋅ Max Memory
#   ⋅ Average Memory
#
# Mails of a batch are exposed in the pwd in a simple text format

echo "Showing results for batch $(pwd)"

for var in "CPU time" "Run time" "Turnaround time" "Max Memory" "Average Memory"
do
        echo $var
        echo "Min     Max     Mean     Std" | column -t
        # Get the var value in all the text file present in the PWD and compute statistics
        cat * | grep $var | awk '{print $4}' | datamash --sort min 1 max 1 mean 1 sstdev 1 | column -t
        echo ""
done
