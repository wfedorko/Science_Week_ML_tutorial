# Science Week Machine Learning hands-on session

## Introduction
This repository holds the notebooks and code for the Machine Learning hands-on session at 2019 Science Week, Data Science and Quantum Computing Workshop on August 23rd. We will explore the application of Convolutional Neural Networks to the problem of particle identification in Water Cherenkov Detector.
Before proceeding please fork this repository by clicking on a button above in top right corner of the page.

## Acknowledgements
I borrowed code liberally from [code and tutorials](https://github.com/WatChMaL) developed by [Kazu Terao](https://github.com/drinkingkazu) and code by [Julian Ding](https://github.com/search?q=user%3Ajulianzding) and [Abhishek Kajal](https://github.com/search?q=user%3Aabhishekabhishek). Big thanks also to the [Water Cherenkov Machine Learning](https://github.com/WatChMaL) collaboration for lending their data - particularly [Nick Prouse](https://github.com/nickwp) for actually running the simulations and to Julian for 'massaging' the data.
Big Thanks to Amazon Web Services for providing the computing resources enabling us to run this session


<a href="https://aws.amazon.com/what-is-cloud-computing"><img src="https://d0.awsstatic.com/logos/powered-by-aws.png" alt="Powered by AWS Cloud Computing"></a>


## Starting up on AWS instance
Log into your instance. username for everybody is `ubuntu`:
```
ssh -Y -i <path/my_private_key> ubuntu@<aws_instance_assigned_to_me>
```
Then launch a screen/tmux session. Next clone your repository, set up pytorch environment and launch jupyter notebook server. Instructions on how to set up ssh tunnel and bring up the jupyter root screen will be printed on your terminal.
```
screen
git clone <your forked repo url> Science_Week_ML_tutorial
. anaconda3/bin/activate pytorch_p36
cd Science_Week_ML_tutorial
. find_this_ip
./start_jupyternotebook.sh
```
If the instructions do not appear wait 10 seconds and then type:
```
python print_instructions.py
```

## Notebook order in the tutorial
The sequence of the tutorial is:
  1. `Data_Exploration_And_Streaming.ipynb`
  1. `MLP_CNN.ipynb`
  1. `Training diagnostics and performance metrics.ipynb`
The notebook `Training monitor.ipynb` is meant to display some live diagnostics during network training process and can be run anytime in parallel.

