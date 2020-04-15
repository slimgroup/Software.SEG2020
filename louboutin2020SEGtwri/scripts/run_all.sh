#!/bin/bash -l

all=$1
if [ -z $all ]; then
    GL="Y"
    BG="Y"
elif [ $all == "BG" ]; then
    BG="Y"
    GL="N"
elif [ $all == "GL" ]; then
    BG="N"
    GL="Y"
fi

# Data for gaussian lens
if [ -f data/GaussLens_data_tti.jld ] || [ $GL == "N" ] ; then
    echo "Data for Gaussian lens already exist or not wanted, skiping"
else
    echo "Generating data for Gaussian lens"
    julia data/gen_data_GaussLens.jl
fi

# Data for BG compass
if [ -f data/BGCompass_data_tti.jld ] || [ $BG == "N" ] ; then
    echo "Data for BG compass already exist or not wanted, skiping"
else
    echo "Generating data for BG compass"
    julia data/gen_data_bg.jl
fi

# Run Gaussian lens
if [ $GL == "Y" ] ; then
    echo "Running gradient computation for the Gaussian lens"
    julia scripts/gaussian_lens.jl
    echo "Plotting results for the Gaussian lens"
    julia scripts/plot_gaussian_lens.jl
fi

# RunBG Compass
if [ $GL == "Y" ] ; then
    echo "Running gradient computation for the BG compass"
    julia scripts/bg_compass.jl
    echo "Plotting results for the BG Compass"
    julia scripts/plot_bgcompass.jl
fi