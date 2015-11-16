#!/bin/sh

FL=FaceList.csv;
touch $FL;
rm $FL;

K=0;
for dir in ???
do
  cd $dir; 
  for file in *
  do
    echo "$PWD/$file;$K" >> ../$FL;
  done
  let K+=1;
  cd ..
done
