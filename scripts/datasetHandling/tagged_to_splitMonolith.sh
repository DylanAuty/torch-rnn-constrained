#!/bin/bash
# Script that will take a dataset of the format output by "tagAndCopy.sh" and create monoliths in the proportions required.
# User specifies percentage proportions of the dataset to use for training, testing and validation respectively.
# Script will divide the dataset accordingly and create monoliths of null-separated DATA/FORECAST pairs.
# It will place these in an output directory with a sensible name in the place specified.
#
# Directory structure that script will take as input will be:
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
# Script expects 2 arguments:
#	1) The path to the root of the dataset (/path/to/weatherDataProcessed/)
#	2) The path to the root of the intended output.	(/path/to/outputrootdir/MONOLITH_50_30_20/)
# 3) The split of data in percentage form, e.g. -t 50 -s 30 -v 20 for a training-testing-validation split of 50/30/20
#		TOTAL 5 arguments.
# Dylan Auty, 16/05/2016

usage() { echo "Usage: $0 [-i /full/path/to/weatherDataProcessed] [-o /full/path/to/output/dir] [-t <training split, 0-100>] [-s <testing split, 0-100>] [-v <validation split, 0-100>]" 1>&2; exit 0; }

while getopts ":i:o:t:s:v:" a; do
	case "${a}" in
		i)
			i=${OPTARG}
			;;
		o)
			o=${OPTARG}
			;;
		t)
			t=${OPTARG}
			;;
		s)
			s=${OPTARG}
			;;
		v)
			v=${OPTARG}
			;;
		h)
			usage
			;;
		*)
			echo "ERROR: Invalid argument"
			usage
			;;
	esac
done

# Sanity checking arguments
if [ $((t+s+v)) != 100 ]; then
	echo "ERROR: Proportions given sum to $[t+s+v], must sum to 100.";
	usage
fi
if [ $OPTIND != 11 ]; then
	echo "ERROR: Incorrect number of arguments. expecting exactly 5.";
	usage
fi
if ! [ -d ${i} ]; then
	echo "ERROR: Argument following '-i' is not a directory.";
	usage
fi
if ! [ -d ${o} ]; then
	echo "ERROR: Argument following '-o' is not a directory.";
	usage
fi

start_dir=$(pwd)

# Create the output directory/file names.
outFolder="weatherDataMonolith-${t}-${s}-${v}"
trainFileName="trainMonolith-${t}-${s}-${v}.txt"
testFileName="testMonolith-${t}-${s}-${v}.txt"
validateFileName="validateMonolith-${t}-${s}-${v}.txt"

# Count the number of files in one of the folders and add one to make the count accurate
fileCount=$(($(ls -1 ${i}/rawData | wc -l) + 1))
# Work out the file numbers to stop at for every fraction t, s and v
# Note that this script will round any fractions down.
centileCount=$(( fileCount/100 ))
t_lim=$((${t}*$centileCount))
s_lim=$(($t_lim + ${s}*$centileCount))
v_lim=$(($s_lim + ${v}*$centileCount))

# Create the output directory where specified.
mkdir ${o}/$outFolder

for it in `seq 1 $fileCount`; do	# Iterate over every file, files are numbered 1 - fileCount so this will work.
	if (( $it <= $t_lim )); then # Train proportion
		cat	${i}/rawData/$it.data >> "${o}/$outFolder/$trainFileName"
		echo "" >> "${o}/$outFolder/$trainFileName"
		cat "${i}/referenceForecasts/$it.txt" >> "${o}/$outFolder/$trainFileName"
		echo $'\u0000'	>> "${o}/$outFolder/$trainFileName"	# This should be a null character
	elif (( $it > $t_lim && $it <= $s_lim )); then # Test proportion
		cat	${i}/rawData/$it.data >> "${o}/$outFolder/$testFileName"
		echo "" >> "${o}/$outFolder/$testFileName"
		cat "${i}/referenceForecasts/$it.txt" >> "${o}/$outFolder/$testFileName"
		echo $'\u0000'	>> "${o}/$outFolder/$testFileName"	
	elif (( $it > $s_lim && $it <= $v_lim )); then # Validate proportion
		cat	${i}/rawData/$it.data >> "${o}/$outFolder/$validateFileName"
		echo "" >> "${o}/$outFolder/$validateFileName"
		cat "${i}/referenceForecasts/$it.txt" >> "${o}/$outFolder/$validateFileName"
		echo $'\u0000'	>> "${o}/$outFolder/$validateFileName"
	fi
done

