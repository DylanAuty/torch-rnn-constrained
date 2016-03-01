#!/bin/bash
# Script to concatenate all pairs of *.events and *.text, corresponding to the raw weather data and the text forecast respectively, together.
# Initial version will only work for the weathergov dataset, may change this in the future to be more general.
# Format will be:
#	
#	* ===== BEGIN DATA ===== *
#	{DATA}
#	
#	* === BEGIN FORECAST === *
#	{FORECAST}
#	
#	* ======== END ========= *
#	\0
#
# Headers/footers exist only for readability and are liable to be changed in future.

if [ $# != 1 ]; then
	echo "ERROR: Please provide path to root directory to search"
	exit 1
else
	# Note - use find depth to get list of folders, then xargs to a for loop for each folder?
	find "$1" -mindepth 2 -maxdepth 2 -print0 | xargs -0 -I '{}' sh -c "for FILE in {}/*.text; do
		BASE=${FILE%.text};
		"

	#find "$1" -name "*.text" -print0 | xargs -0 -I '{}' sh -c "cat {} >> concatOutput.txt ; cat endForecast.txt >> concatOutput.txt"
fi

