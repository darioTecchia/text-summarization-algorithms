# Text Summarization

## Introduction

This repository contains scripts about automatic text synthesis and related methods to measure the goodness of the algorithmically obtained summary.

## Project Strucure
Inside the **extractive** and **abstract** folders are all the extractive and abstract summaries, respectively.

The **datasets** folder contains all datasets used during the experiments (csv files must be downloaded separately).

The **evaluation** folder contains the scripts that apply the evaluation algorithms to the machine-generated summaries. At the moment, there are two scripts:
- `evaluation.bert.py`: run the bert evaluations for each computer-obtained summary
- `evaluation.rouge.py`: run the rouge evaluations for each computer-obtained summary

The **outputs** folder is the outsource folder that contains the csv files obtained with the scripts.

The **runner** folder contains all scripts that apply the summary algorithms to the dataset. There are three scripts:
- *run.all.abstractive*: run all the abstractive algorithms 
- *run.all.extractive*: run all the extractive algorithms 
- *run.all*: run all the algorithms 

## Usage
First of all install all the libraries required by python with the command `pip3 install -r requirements.txt`.

Download the dataset from here: **AGGIUNGI LINK QUI**.

After all, launch the scripts contained into the folders.

## Text Summarization Algorithms
In the following table are listed all the summarization algorithms used during the experiments.

| Algorithm | Approach    | Kind                          |
|-----------|-------------|-------------------------------|
| BERT      | Extractive  |                               |
| KL_SUM    | Extractive  |                               |
| LEXRANK   | Extractive  |                               |
| LSA       | Extractive  |                               |
| LUHN      | Extractive  | Frequency-Based Summarization |
| TEXTRANK  | Extractive  |                               |
| T5        | Abstractive |                               |
| BART      | Abstractive |                               |

## Sumary Evaluation Algorithms
In the following table are listed all the evaluation algorithms used during the experiments.

`// TODO`