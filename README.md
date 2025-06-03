# Legal Document Topic Modeling: Analyzing Judicial Decisions with NLP

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project applies advanced Natural Language Processing (NLP) techniques to analyze and model topics in Brazilian legal documents, focusing on lawsuits related to special education policies. Using a corpus of 4,259 judicial cases from the São Paulo Court of Justice, the system uncovers latent thematic structures through probabilistic topic modeling methods including Latent Dirichlet Allocation (LDA), Latent Semantic Analysis (LSA), and Probabilistic Latent Semantic Analysis (pLSA).

The study evaluates these models quantitatively via coherence scores and qualitatively through human interpretability assessments, highlighting the strengths and limitations of each approach in the legal domain. This work demonstrates how AI can enhance legal transparency, accessibility, and analytical depth without compromising legal interpretative autonomy.

## Features

- **Data Processing**: Cleans and preprocesses legal documents for analysis
- **Exploratory Analysis**: Visualizes document distributions, legal subjects, and procedural classes
- **Topic Modeling**: Implements and compares multiple topic modeling techniques
- **Model Evaluation**: Uses coherence scores to evaluate and select optimal models
- **Visualization**: Generates interactive visualizations of topics and their distributions

## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/Etore-BeS/interpreting-lawsuits-topic-modeling.git](https://github.com/Etore-BeS/interpreting-lawsuits-topic-modeling.git)
   cd interpreting-lawsuits-topic-modeling

   ```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
python -m spacy download pt_core_news_sm
```

Project Structure
├── data/               # Data files (raw and processed)
├── notebooks/          # Jupyter notebooks for analysis
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   └── 03_topic_modeling_training.ipynb
├── src/                # Source code
├── models/             # Trained models
├── requirements.txt    # Project dependencies
└── README.md          # This file

## Usage

### Data Preparation:

Place your legal documents in the data/raw/ directory
Run the preprocessing notebook to clean and prepare the data

### Exploratory Analysis:

Open and run 02_exploratory_data_analysis.ipynb to explore the dataset

### Topic Modeling:

Run 03_topic_modeling_training.ipynb to train and evaluate topic models
Adjust parameters in the notebook as needed

### Results

- pLSA excels at capturing broad legal themes with fewer topics.
- LDA performs best at higher topic counts, distinguishing nuanced legal issues such as educational access and contractual disputes.
- LSA shows limitations in handling complex legal language and thematic granularity.
- Human evaluation confirms the interpretability and relevance of topics, complementing quantitative coherence metrics.

## Dependencies

Based on the [requirements.txt](requirements.txt) file, this project requires:

- Python 3.11+
- Key packages:
  - spaCy (version 3.7.5)
  - Portuguese language model (version 3.7.0)
  - pyLDAvis (version 3.4.1)
  - Folium (version 0.19.4)
  - Pandas (version 2.2.2)
  - NumPy (version 1.26.4)
  - Gensim (version 4.3.3)
  - scikit-learn (version 1.6.1)
  - Plotly (version 5.18.0)
  - NLTK (version 3.8.1)
  - Jinja2 (version 3.1.5)
  - python-dateutil (version 2.8.2)
  - joblib (version 1.4.2)
  - Matplotlib (version 3.7.2)
  - Seaborn (version 0.13.2)

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- This research was supported by the University of São Paulo  undergraduate grant.
- spaCy for Portuguese NLP
- Gensim for topic modeling
- Plotly for interactive visualizations
