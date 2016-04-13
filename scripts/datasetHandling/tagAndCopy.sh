#!/bin/bash
# Script to give each pair of *.events and *.text a numerical ID, while copying them to a new dataset file structure
# This is done as searching through the file directory is a bit of a pain at the moment, and this will make it easier to
# split the dataset up into training/testing/validation sets.
#
# Directory structure will be:
#	
# weatherDataProcessed
# ├── rawData
# │   ├── 000001.data
# │   ├── 000002.data
# │   └── [etc.]
# ├── referenceForecasts
# │   ├── 000001.txt
# │   ├── 000002.txt
# │   └── [etc.]
# ├── dataForecastPairs
# │   ├── 000001.pair
# │   ├── 000002.pair
# │   └── [etc.]
# ├── Experiment_1
# ├── Experiment_2
# ├── Experiment_3
# └── [etc.]
# 
# By processing this data in one go and storing a copy of the files, it is hoped that an overhead may be
# removed when running experiments.
#
# data/Forecast pairs will be exported in JSON format for ease of access
# This also allows me to check that the script is tagging things properly.
#
# Script expects 2 arguments:
#	1) The path to the root of the dataset
#	2) The path to the root of the intended output.
#
# Dylan Auty, 09/04/2016


function cpToStructure(){
	for FILE in $1/*.text; do
		fileCounter=`cat ./tmp.dat`
		BASE=${FILE%.text}
		echo "$fileCounter: $BASE"
		cp $BASE.events ./rawData/$fileCounter.data
		cp $BASE.text ./referenceForecasts/$fileCounter.txt
		echo '{"data": "' >> ./dataForecastPairs/$fileCounter.pair
		cat $BASE.events >> ./dataForecastPairs/$fileCounter.pair
		echo '", "text" : "' >> ./dataForecastPairs/$fileCounter.pair
		cat $BASE.text >> ./dataForecastPairs/$fileCounter.pair
		echo '"}' >> ./dataForecastPairs/$fileCounter.pair	
		((fileCounter++))
		echo $fileCounter > ./tmp.dat
	done
}

function makeStructure(){
	echo "Creating directory structure in $1..."
	mkdir $1/weatherDataProcessed
	cd $1/weatherDataProcessed
	mkdir rawData
	mkdir referenceForecasts
	mkdir dataForecastPairs
	echo "Done creating directory structure."
}

export -f cpToStructure
export -f makeStructure

if [ $# != 2 ]; then
	echo "ERROR: Please provide 2 arguments:"
	echo "1) The path to root data directory to search"
	echo "2) The path for where to put the output file structure."
	exit 1
else
	# First generate the new directory structure

	makeStructure $2
	cd $2/weatherDataProcessed
	# Need this to move us into the correct directory before cpToStructure starts putting things in their places.
	
	# Use xargs to fetch every file basename, then pass it to cpToStructure to be filed away.
	# Note - use find depth to get list of folders, then xargs to a for loop for each folder
	# Because of xargs and scope issues from it, the counter for naming has to be stored in a file.

	echo 0 > ./tmp.dat

	find "$1" -mindepth 2 -maxdepth 2 -print0 | xargs -0 -I '{}' bash -c 'cpToStructure {}'

	rm ./tmp.dat
fi



