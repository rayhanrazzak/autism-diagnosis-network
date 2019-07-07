# autism-diagnosis-network

## Background

Autism spectrum disorder (ASD) is a disability that affects 1 in 59 Americans [1]. Current methods of diagnosis rely on subjective behavioral examinations that lead to false-positives and false-negatives **citation needed**. Thus, the need for a more accurate and reliable methodology is prevalent.

[1] https://www.cdc.gov/mmwr/volumes/67/ss/ss6706a1.htm

## About the Project

We propose the use of machine learning algorithms to analyze patient's magnetic resonance imaging (MRI) scans in order to properly diagnose ASD in children aged 6-10.

Since it is uncertain whether functional MRIs (fMRIs) or anatomical MRIs have a stronger correlation to ASD, we present two convolutional neural networks (CNN) trained via magnetic resonance imaging **citation needed**.    

The first is a three-dimensional CNN trained using anatomical MRIs.

The second is a four-dimensional spatio-temporal CNN (ST-CNN) trained using fMRIs. ST-CNNs have never been used for medical diagnosis. Thus, we implement this novel neural network because fMRIs track brain functionality over time. Therefore, the algorithm needs to study the MRI image's change with relation to time.  


## Notes
* ST-CNN refers to CNNs capable of analyzing data patterns of time.
* MRI datasets were obtained from [Autism Brain Imaging Data Exchange (ABIDE).](http://fcon_1000.projects.nitrc.org/indi/abide/)
* Part of a submission to the 2019 Google Science Fair.
* Part of a submission to [MIT THINK Scholars Program. ](https://docs.google.com/document/d/12GTQrbj2fcYvDgfmMtAV4yThOQrYORfl6feBI63tM30/edit?usp=sharing)
