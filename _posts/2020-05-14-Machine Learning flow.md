---
title:  "Supervised Machine Learning"
date: 2020-05-14
categories: [Reading]
tags: [machine Learning,data science]
excerpt: "Read about the Supervised machine learning workflow"
author_profile: true
mathjax: true
---

The entire workflow of any supervised machine learning project can be described via 8 steps shown below:  

![Machine learning flow](https://github.com/muzammil-iftikhar/muzammil-iftikhar.github.io/blob/master/assets/images/Machine-learning-steps.JPG)

We will divide our every Data science & supervised Machine learning project into these eight simple steps.  
We will get the real taste of these steps in future tutorials where we will see each step in practice

**Define Data**  
Before proceeding with any project, you first define the problem you are going to address. Is it a regression problem or a classification one ?

**Acquire Data**  
You acquire your data from different sources. For most of our projects, we will be acquiring our dataset from online resources such as Kaggle

**Import Data**  
We will import our data into python with Pandas

**Data cleaning, Data completing, Feature Engineering**  
We will be cleaning our data by dropping the features that are not needed (if there is any)  
We will be completing our data by eiter filling in the NAN/null values or dropping them altogether  
We may need to create new features based on the existing feature set where required

**Get Model**  
We will get our model and instantiate it

**Train Model**  
You train your model with the training dataset

**Test Model**  
You test your model with the test dataset

**Model Validation**  
Since in each project, your target is to predict something. Obviously, in the end, we would like to know, how well did our model do ? In this step, you will validate your model and if the validation results are not satisfactory, you may either retrain your model with different parameters or you will get a different mode
