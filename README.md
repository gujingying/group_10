# Forest Cover Type Classification

## Introduction

## Models
### Linear Discriminant Analysis
### Random Forest Classifier
### Support Vector Machine
### K-Nearest Neighbours

## Performance

## Environment

## How to run
Using command line to run the script.

### Command Syntax

```
python <DATA FILE> <COMMAND LINE OPTIONS/FLAGS>
```

### Options

- *-m --model*: Specify which model to run.

  parameters: knn; rfc; lda; svm; all

  default: all


- *-l --location*: Specify the location of the data set.

  default: covtype.csv


### Example

```commandline
python -l covtype.csv -m rfc
```


