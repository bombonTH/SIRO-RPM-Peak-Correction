# SIRO-RPM-Peak-Phase-Correction #
Siriraj Radiation Oncology department's script for converting Varian's .vxp file to .csv for research use.
Featured in "Accuracy of Target Volume and Artifact reduction by Optimal Sorting Methods of 4DCT image reconstruction on Lung Cancer Radiotherapy in Patients with a mismatched pitch in irregular respiration"
The script can also:
* Redetect peak and valley for 4DCT imaging reconstruction.
* Apply lowpass filter to clean up noise
* Calculate basic respiratory statistics
* Plot the signal before and after correction.

## Requirements: ##
```
matplotlib==3.5.1
numpy==1.22.1
peakdetect==1.2
scipy==1.7.3
```

## Usage: ##
```
vxp2csv.py
--file <filename>
--all to read all .vxp files in ./data/in directory (Automatically True when no --file is supplied)
--lowpass to apply lowpass filter before peak and phase detection
--no-phase-correction to disable automatic peak and phase detection
--plot to plot the resulting signal
```
