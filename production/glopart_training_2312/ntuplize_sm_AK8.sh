#!/bin/bash -x

JOBNUM=$1
JOBBEGIN=$2
FILELIST=$3 #samples/sm_all_filelist.txt
MACHINE=$4
SCRIPT_NAME=$5
if [[ -z $SCRIPT_NAME ]]; then
    SCRIPT_NAME=infer_sophon.C
fi

JOBNUM=$((JOBNUM+JOBBEGIN))


# using the jobnum to read the line, DIR is current directory
DIR="$( cd "$( dirname "$0" )" && pwd )"
FILEIN=$(sed -n "$((JOBNUM+1))p" "$DIR/$FILELIST")
if [[ "$FILEIN" == "" ]]; then
    echo "No more files to process"
    exit 1
fi

#extract samples/proc
FIRST_LINE=$(head -n 1 $FILELIST)
PROC=$(echo "$FIRST_LINE" | awk -F'/' '{print $1 "/" $2}')
echo "Extracted proc: $PROC"

# basic configuration
if [[ $MACHINE == "ihep" ]]; then
    INPUT_PATH=/publicfs/cms/user/$USER/condor_output
    OUTPUT_PATH_TUPLE=/publicfs/cms/user/$USER/ntuplizer/output
    DELPHES_PATH=/publicfs/cms/user/zhaoyz/utils/Delphes-3.5.0
    LOAD_ENV_PATH=/scratchfs/cms/licq/utils/load_standalonemg_env.sh
    NTUPLIZER_FILE_PATH=/publicfs/cms/user/zhaoyz/ntuplizer/sophon/analyzers
    MODEL_NAME=JetClassII_Sophon.onnx
fi

## load environment
if [ ! -z "${CONDA_PREFIX}" ]; then
    conda deactivate
fi
echo "Load env"

#source /cvmfs/cms.cern.ch/cmsset_default.sh
#source $LOAD_ENV_PATH

source /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-el9-gcc13-opt/setup.sh
export ROOT_INCLUDE_PATH=$ROOT_INCLUDE_PATH:/cvmfs/sft.cern.ch/lcg/releases/delphes/3.5.1pre09-9fe9c/x86_64-el9-gcc13-opt/include:/cvmfs/sft.cern.ch/lcg/releases/onnxruntime/1.15.1-8b3a0/x86_64-el9-gcc13-opt/include/core/session


RANDSTR=$(tr -dc A-Za-z0-9 </dev/urandom | head -c 10; echo)
#WORKDIR=$OUTPUT_PATH_TUPLE/workdir_$(date +%y%m%d-%H%M%S)_${RANDSTR}_$(echo "$PROC" | sed 's/\//_/g')_$JOBNUM
WORKDIR=/tmp/$USER/workdir_$(date +%y%m%d-%H%M%S)_${RANDSTR}_$(echo "$PROC" | sed 's/\//_/g')_$JOBNUM
mkdir -p $WORKDIR

cd $WORKDIR

cp -r $NTUPLIZER_FILE_PATH/* .

# process the paths
FILEIN_PATH=$INPUT_PATH/$FILEIN

## extract the dir name
DIR_PATH=$(dirname "$FILEIN_PATH")
FILE_NAME=$(basename "$FILEIN_PATH")

OUTPUT_PATH_TUPLE_PROC=$OUTPUT_PATH_TUPLE/$PROC
mkdir -p $OUTPUT_PATH_TUPLE_PROC
NTUPLE_NAME=ntuple_$FILE_NAME

# run ntuplizer
root -b -q $SCRIPT_NAME'++("'$DIR_PATH/$FILE_NAME'", "'$OUTPUT_PATH_TUPLE_PROC/$NTUPLE_NAME'", "'$MODEL_NAME'")'

# remove workspace
rm -rf $WORKDIR
