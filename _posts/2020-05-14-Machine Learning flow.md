---
title:  "Supervised Machine Learning"
date: 2020-05-14
categories: [Reading]
tags: [machine Learning,data science]
excerpt: "Read about the Supervised machine learning workflow"
author_profile: true
mathjax: true
---

The entire workflow of any supervised machine learning project can be described in 8 steps shown below:  

![Machine learning flow](/assets/images/machine_learning_flow.JPG)

We will divide our every Data science & supervised Machine learning project into these eight simple steps.  
You will get the real taste of these steps in future tutorials where we will see each step in practice so don't worry or stress on them too much for now. Just get an idea of what each step means and in future tutorials i will reference this page alot when we put each and every step in practice.

**Define Problem**  
Before proceeding with any project, you first define the problem you are going to address. Is it a regression problem or a classification one ?

**Acquire Data**  
You acquire your data from different sources. For most of our projects, we will be acquiring our datasets from online sources such as Kaggle

**Import Data**  
We will import our data into python with Pandas

**Data cleaning, Data completing, Feature engineering**  
We will be cleaning our data by dropping the features that are not needed (if there is any)  
We will be completing our data by either filling in the Nan/null values or dropping them altogether  
We may need to create new features based on the existing feature set where required

**Get Model**  
We will get our model and instantiate it

**Train Model**  
We will train our model with the training dataset

**Test Model**  
We test our model with the test dataset

**Model Validation**  
Since in each project, our target is to predict something. Obviously, in the end, we would like to know, how well did our model do ? In this step, we will validate our model and if the validation results are not satisfactory, we may either retrain our model with different parameters or we will get a different model
