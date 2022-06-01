# -*- coding: utf-8 -*-
"""
Created on Sat May 11 09:49:18 2019

@author: Serge
"""

import os
from pathlib import Path
import subprocess
import shutil


# Function to put all files into their own folder:
def sort_files_into_folders(top):
    os.chdir(top)
    # Return list of files in folder:
    filenames = list(os.walk(os.getcwd()))[0][2]
    for f in filenames:
        os.mkdir(top+'/'+os.path.splitext(f)[0])
        shutil.copy2(top+'/'+f, top+'/'+os.path.splitext(f)[0])
    return

### Uncomment the 2 lines below if you want to put each file in a folder into its
### own folder (of the same name as the file).
#top = input('folder containing files:\n>')
#sort_files_into_folders(top)
# Check that all folders have 1 file in them; should return "True":
#filenames = list(os.walk(os.getcwd()))[0][2]
#np.array([(len(f[2]) == 1) for f in filenames]).all()

###
    
# Get the working directory of this script:
current_script_dir = Path(__file__).parent.absolute()
scipt_to_use_fname = Path.joinpath(current_script_dir, r'sct1-1_angle measurement script 2020-05-17.py')

# Run "sct1-1_angle measurement script 2020-05-17" on all subdirectories in 
# folder:
# Note that sct1-1_angle measurement script 2020-05-17 needs to be in the same
# directory/folder as this script
top = input('folder containing files:\n>')

folders = list(os.walk(top))[0][1]
for f in folders:
    #use top+f as first input in script:
    process = subprocess.Popen(['python', str(scipt_to_use_fname)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)

    process.stdout.readline()
    process.stdout.readline()
    # pass the angle measurement script the directory+filename:
    process.stdin.write(top+'\\'+f+'\n')
    # pass the angle measurement script the number of pixels to use for making
    # angles (I used 9 for all the live reaggregation data):
    process.stdin.write('9\n')
    # pass the angle measurement script the number of pixels to use for making
    # linescans (I used 12 for all the live reaggregation data):
    process.stdin.write('12\n')
    # pass the angle measurement script the number of pixels to use for making
    # curvature measurements (I used 9 for all the live reaggregation data):
    process.stdin.write('9\n')
    # use the default disk radius of the script (16px):
    process.stdin.write('\n')
    # use the default background threshold (10% of the dtype max):
    process.stdin.write('\n')
    # use the default object brightness threshold (20% of the max dtype):
    process.stdin.write('\n')
    # Yes, save the output:
    process.stdin.write('y\n')
    # use the default filename and location to save ouput:
    process.stdin.write('\n')
    
    # save over the previous files in the directory:
    process.stdin.write('y\n')
    # The above line can be commented out if images have not yet been analyzed.
    
    # press enter to quit:
    process.stdin.write('\n')
    process.stdin.flush()
    process.stdin.close()
    #process.wait(timeout=360)
    #for i in process.stdout:
    #    print(i)
    
    while True:
        line = process.stdout.readline()
        if len(line) == 0:
            break
        print(line)
    



