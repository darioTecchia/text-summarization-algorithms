# Text Summarization
[![Python 3.9.9](https://img.shields.io/badge/python-3.9.9-blue.svg)](https://www.python.org/downloads/release/python-399/)

## Introduction

This repository contains scripts about automatic text synthesis and related methods to measure the goodness of the algorithmically obtained summary.

## Project Strucure
Inside the **extractive** and **abstract** folders are all the extractive and abstract summaries, respectively.

The **datasets** folder contains all datasets used during the experiments (csv files must be downloaded separately).

The **evaluation** folder contains the scripts that apply the evaluation algorithms to the machine-generated summaries. At the moment, there are two scripts:
- `evaluation_bert.py`: run the bert evaluations for each computer-obtained summary
- `evaluation_rouge.py`: run the rouge evaluations for each computer-obtained summary
- `evaluation_bleu.py`: run the bleu evaluations for each computer-obtained summary

The **outputs** folder is the outsource folder that contains the csv files obtained with the scripts.

The **runner** folder contains all scripts that apply the summary algorithms to the dataset. There are three scripts:
- *runner.py*: summarization algorithms runner

## Usage
First of all install all the libraries required by python with the command `pip3 install -r requirements.txt`.

Download the dataset from [here](https://drive.google.com/file/d/1iAcqK6sOK_kMYJn4kptVJA2QpTJoqk--/view?usp=sharing) and move it into the **datasets** folder.

You can also download the ScisummNet dataset from [here](https://cs.stanford.edu/~myasu/projects/scisumm_net/scisummnet_release1.1__20190413.zip) and after you can convert it with the `top1000.converter.py` script.

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