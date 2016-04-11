#!/bin/bash
# Script to concatenate all the files ending with the extension ".text" in all subdirectories of a specified folder

if [ $# != 1 ]; then
	echo "ERROR: Please provide path to root directory to search"
	exit 1
else
	find "$1" -name "*.text" -print0 | xargs -0 -I '{}' sh -c "cat {} >> concatOutput.txt ; cat endForecast.txt >> concatOutput.txt"
fi

