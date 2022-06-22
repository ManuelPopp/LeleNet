#!/bin/bash 
AllExp=tensorboard dev list | grep https://tensorboard.dev/experiment/ | sed "s|https://tensorboard.dev/experiment/||g" | tr '\n' ',' | sed 's/\///g'
tensorboard dev delete $AllExp
