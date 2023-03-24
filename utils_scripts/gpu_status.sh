#! /bin/bash

getGB() {
  awk '{print(int($1/1024))}'
}

getGpusOcc() {
  node=$1
  cnt=0
  for jobId in $(squeue | grep -P $node'$' | awk '{print $1}')
  do
    gpus=$(scontrol show job -dd $jobId | grep TresPerNode | cut -d= -f2 | cut -d: -f2)
    ## echo "    getGpusOccup |$node|$jobId|$gpus|" 1>&2
    if test ! -z "$gpus"
    then
      cnt=$(expr $cnt + $gpus)
    fi
  done
  echo $cnt
}

tmpF=/tmp/Dsgs.$$

## scontrol -o show nodes | grep gpu-all > $tmpF
scontrol -o show nodes | egrep '(gpu-all|cliffjumper)' > $tmpF

printf "%8s %8s %8s %24s %26s\n" Node State Cores RAM GPUs
printf "%8s %8s %8s %8s %8s %8s %8s %8s %6s %6s %6s\n" ' ' ' ' tot used free totGB usedGB freeGB tot used free

while read entry
do
  nname=$(echo $entry | tr ' ' '\012' | grep NodeName | cut -d= -f2)

  state=$(echo $entry | tr ' ' '\012' | grep State | cut -d= -f2)

  cpuTot=$(echo $entry | tr ' ' '\012' | grep CPUTot | cut -d= -f2)
  cpuOcc=$(echo $entry | tr ' ' '\012' | grep CPUAlloc | cut -d= -f2)
  cpuFree=$(expr $cpuTot - $cpuOcc)

  ramTot=$(echo $entry | tr ' ' '\012' | grep RealMemory | cut -d= -f2 | getGB)
  ramOcc=$(echo $entry | tr ' ' '\012' | grep AllocMem | cut -d= -f2 | getGB)
  ramFree=$(expr $ramTot - $ramOcc)

  gpuTot=$(echo $entry | tr ' ' '\012' | grep Gres | cut -d= -f2 | cut -d: -f3)
  gpuOcc=$(getGpusOcc $nname)
  gpuFree=$(expr $gpuTot - $gpuOcc)

  printf "%8s %8s %8d %8d %8d %8d %8d %8d %6d %6d %6d\n" $nname $state $cpuTot $cpuOcc $cpuFree $ramTot $ramOcc $ramFree $gpuTot $gpuOcc $gpuFree
done < $tmpF

\rm $tmpF

exit
