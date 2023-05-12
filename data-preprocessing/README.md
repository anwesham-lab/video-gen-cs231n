## Prerequisite
| **Package**    | **version**  |
|----------------|--------------|
| python         |  >=3.5       |
| pytorch        |  1.12        |
| numpy          |  1.17.2      |
| pandas         |  0.25.1      |
| tensorboardX   |  1.8         |
| ffmpeg	 |  3.4.2	|


## Dataset Prep
'''
brew install ffmpeg #or sudo apt install ffmpeg
chmod u+x scripts/*.sh
scripts/download_ucf.sh data/UCF101
scripts/download_annotations.sh data/UCF101
'''