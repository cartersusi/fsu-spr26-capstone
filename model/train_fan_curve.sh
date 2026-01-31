#!/bin/bash

cleanup() {
    echo "Restoring automatic fan control..."
    sudo nvidia-settings -a "[gpu:0]/GPUFanControlState=0"
    exit
}

trap cleanup EXIT ERR INT TERM

sudo nvidia-settings -a "[gpu:0]/GPUFanControlState=1"
while true; do
    TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)
    if [ -z "$TEMP" ] || ! [[ "$TEMP" =~ ^[0-9]+$ ]]; then
        echo "Error reading temperature"
        exit 1
    fi
    if [ "$TEMP" -lt 50 ]; then SPEED=40
    elif [ "$TEMP" -lt 60 ]; then SPEED=50
    elif [ "$TEMP" -lt 70 ]; then SPEED=70
    elif [ "$TEMP" -lt 75 ]; then SPEED=80
    else SPEED=100
    fi
    sudo nvidia-settings -a "[fan:0]/GPUTargetFanSpeed=$SPEED" || exit 1
    sleep 5
done