Hello,
If you are reading this, thank you for taking a deeper interest in my scientific work. This file describes some of the other files in the 'misc' folder of my data analysis.

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

Second, the excel files 'adh_kin_checklist.xlsx' and 'lifeact_quant_checklist.xlsx' were used as a checklist as I made progress through analyzing data. There are occasionally notes for some of the files that were analyzed.

Third, when creating the vectors used to determine the contact angle pixels are counted. The number of pixels to move away from the contact edge has an optimal value depending on the size of the cell being analyzed. The .pdf files 'optimized pathfinder length fig - area.pdf' and 'optimized pathfinder length fig - radius.pdf' do a good job of communicating what these optimum values are I think. They are essentially indicated by the red line.

Lastly, in order to make the image resolution acquired from the Zeiss Axiovert 200M with a 20X objective as comparable to the data acquired with the Leica SP8 with a 40X water immersion objective an equation relating the physical dimensions of a single pixel in an image to the images pixel resolution (###px by ###px) and the digital zoom. The file 'optimum_zoom_and_format_table.html' has a table summarizing these parameters starting with those combinations that deviate the least from the physical dimensions of 1 pixel from an image acquired using the Zeiss Axiovert. (FYI the physical dimensions are ~0.506675079794729 micrometers per pixel in both the x and y directions.) Since not all image acquisition resolutions are possible with the Leica SP8 several of these optimal values were tried with the goal of introducing as little error as possible. The settings with the lowest error was: 
image acquisition resolution = 736pixels x 736pixels
zoom = 0.78

-Serge Parent.