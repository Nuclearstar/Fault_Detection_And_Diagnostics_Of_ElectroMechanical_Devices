# Fault Detection & Diagnostics Of ElectroMechanical Devices

## Overview:

Fault Detection & Diagnostics of ElectroMechanical Devices is a project based on Sinfonia conveyor belts used in airports. Outliers detection, classiﬁcation & prediction were the main tasks involved.

Various statistical tests like Grubbs test, Modified Thompson Tau test were used to find the outliers in the given datasets using Python. Finding outliers separately in the different features of the dataset itself is fault detection and classification. Further a prediction was made using algorithms like SVM, Decision trees to know the next series of data whether faulty or not. If faulty over a run was found then suitable diagnosis was applied to make the machine run smooth.

## Installation Dependencies:

- Python 3
- pandas
- numpy
- scipy

## Implementation

outlierDetect.ipynb file has the detailed solution for finding outliers which helps to detects any faults occuring in conveyor belts.

## How to run?

- First clone the project into your local system
```
git clone https://github.com/Nuclearstar/Fault_Detection_And_Diagnostics_Of_ElectroMechanical_Devices.git
```
- Then change directory to this project
```
cd Fault_Detection_And_Diagnostics_Of_ElectroMechanical_Devices
```
- Then setup a virtual env
```
python -m venv myenv
```
- Then activate your virtual env
```
cd myenv
cd Scripts
activate
```
- Further change directory to project root
```
cd ..
cd ..
```
- Next install all the required packages in the virtual env
```
pip install -r requirements.txt
```
- Now you are ready to run the program
```
jupyter notebook
```
