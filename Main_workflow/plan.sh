#!/bin/sh
format=$(getopt -n "$0" -l "inputs:" -- -- "$@")
if [ $# -lt 1 ]; then
   echo "Wrong number of arguments are passed."
   exit
fi
eval set -- "$format"

#Read the argument values
while [ $# -gt 0 ]
do
     case "$1" in
          --inputs) inputs="$2"; 
     esac
     shift
done
pegasus-plan --conf pegasus.properties \
    --dir submit \
    --sites condorpool \
    --output-sites local \
    --cleanup leaf \
    --force \
    $inputs