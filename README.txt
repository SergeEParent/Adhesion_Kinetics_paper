Hello,
If you are reading this, thank you for taking a deeper interest in my scientific work. This file describes some of the files used in my data analysis.

First and most importantly, the scripts used for measuring cells and for analyzing the data were prepared using Anaconda in a Python3 environment. The file adh_kin_analysis.yml contains the packages and dependencies I used to get my results from my timelapse imaging files. Those imaging files are kept in a separate location.

To make an environment from this yml file you can do the following:
1. install Anaconda
2. open the Anaconda prompt
3. type in "conda env create -f adh_kin_analysis.yml" (without the quotes) and hit enter

To use this environment file:
After step 3 above you can:
4. activate the environment type in "conda activate adh_kin_analysis_env"
the above term 'adh_kin_analysis_env' should be the name of the environment, but this can be checked by opening the yml file in notepad and looking at the first line. If the name is different then just type that one in (or rename it).
Anaconda should be running in this environment now, so the other Python files I made should run identically to how they were run on my machine.

As for the script naming conventions - they are named according to the order in which they are to be run. If you have the dataset that I analyzed you can start from 'sct1-0' and then run the scripts sequentially until 'sct5' and the files should make the necessary folders and outputs for subsequent scripts to run properly. The scripts 1-5 should be kept in the same folder. Also, 'sct5-viscous shell model clean' can run on its own. It is the implementation of the VESM.

Lastly, a technical note. In order to make the image resolution acquired from the Zeiss Axiovert 200M with a 20X objective as comparable to the data acquired with the Leica SP8 with a 40X water immersion objective an equation relating the physical dimensions of a single pixel in an image to the images pixel resolution (###px by ###px) and the digital zoom. The file 'optimum_zoom_and_format_table.html' has a table summarizing these parameters starting with those combinations that deviate the least from the physical dimensions of 1 pixel from an image acquired using the Zeiss Axiovert. (FYI the physical dimensions are ~0.506675079794729 micrometers per pixel in both the x and y directions.) Since not all image acquisition resolutions are possible with the Leica SP8 several of these optimal values were tried with the goal of introducing as little error as possible. The settings with the lowest error was: 
image acquisition resolution = 736pixels x 736pixels
zoom = 0.78



-Serge Parent.
