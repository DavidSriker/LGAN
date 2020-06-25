#!/usr/bin/env bash

# path is: https://drive.google.com/file/d/1Twmp6WhbG2-ZliI_VuRespuQHlbbDIhP/view?usp=sharing
$(mkdir -p "data/")
echo "Downloading Data"
fileid="1Twmp6WhbG2-ZliI_VuRespuQHlbbDIhP"
filename="data.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
unzip -q "data.zip"
$(rm cookie)
