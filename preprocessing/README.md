## Prerequisite
| **Package**    | **version**  |
|----------------|--------------|
| python         |  >=3.5       |
| pytorch        |  1.12        |
| numpy          |  1.17.2      |
| pandas         |  0.25.1      |
| tensorboardX   |  1.8         |
| ffmpeg	     |  3.4.2	    |


## Dataset Prep
```
cd preprocessing
brew install ffmpeg #or sudo apt install ffmpeg or sudo apititude ffmpeg
chmod u+x scripts/*.sh
scripts/download_ucf.sh data/UCF101
# you may have to run commands one at a time for annotations, issues with confirming the link can come up
scripts/download_annotations.sh data/UCF101 #github has these pushed, don't rerun
scripts/prepare_data.sh 
```