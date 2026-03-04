# Transformer Training and Evaluation for Patronising Language Detection

In this project, I built a transformer training and evaluation pipeline for detecting patronising and condescending language directed towards vulnerable communities.

This is a difficult classification problem because patronising language is rarely as explicit as ordinary toxicity or hate speech. A sentence can appear compassionate or supportive while still portraying a person or community as helpless, inferior or lacking agency.

The dataset is also heavily imbalanced. Only approximately 9.5% of the available examples are labelled as patronising. I therefore focused the project on positive-class F1, recall and detailed error analysis rather than relying on overall accuracy.

My final approach combines RoBERTa-large fine-tuning, controlled experiments across random seeds, decision-threshold optimisation and a quality-pruned weighted ensemble.

## Results

I evaluated the final system on the 2,094-example development set.

| Metric | Result |
|---|---:|
| Positive-class precision | 0.579 |
| Positive-class recall | 0.719 |
| Positive-class F1 | **0.641** |
| Accuracy | 0.924 |
| True positives | 143 |
| False positives | 104 |
| False negatives | 56 |
| True negatives | 1,791 |

The published RoBERTa-base benchmark achieved approximately `0.48` positive-class F1 on the development set. My final system reached `0.641`, an improvement of approximately 16 absolute F1 points.

Although the final accuracy was above 92%, accuracy is not particularly informative for this dataset. A model could predict the negative class for every example and still achieve more than 90% accuracy. For that reason, I used positive-class F1 as the main model-selection objective throughout the project.

## Understanding the dataset

I started by examining the structure of the training and development splits.

| Split | No PCL | PCL | Positive rate |
|---|---:|---:|---:|
| Training | 7,581 | 794 | 9.48% |
| Development | 1,895 | 199 | 9.50% |

I looked at class balance, paragraph lengths and long-sequence outliers before training any models. Positive examples were slightly longer on average, although both classes contained a long tail of unusually large paragraphs.

This helped me choose a sequence length that covered most of the dataset without making every training batch unnecessarily expensive.

## Lexical analysis

I calculated class-discriminative unigram and bigram log-ratios to understand which parts of the vocabulary separated the two classes.

This showed that words connected to emotional framing, helplessness and charity were much more common in positive examples. Some of the strongest associations included terms such as "hopeless", "prayer" and language describing people as being "in need".

These patterns were useful, but they also raised an important concern. A model could learn to associate references to vulnerable communities with patronising language without properly understanding the surrounding context.

Because of this, I did not use keyword rules as the main classifier. I instead used contextual transformer models and later examined whether the same lexical shortcuts still appeared in their errors.

## Baseline model

I first reproduced a RoBERTa-base training pipeline. At this stage, my main goal was to verify that the complete process worked correctly:

- Parsing the original dataset
- Reconstructing the official splits
- Tokenising the paragraphs
- Fine-tuning the classifier
- Calculating positive-class metrics
- Exporting development and test predictions

Once the baseline pipeline was working reliably, I moved to RoBERTa-large for the main experiments.

## Controlled seed experiments

I trained RoBERTa-large across several random seeds while keeping the broader configuration consistent.

I chose this approach because the dataset contains only 794 positive training examples. With a relatively small positive class, changes in parameter initialisation and minibatch ordering can noticeably affect which borderline examples the model learns to recognise.

Rather than choosing one successful run and treating it as representative, I compared the candidate models under the same evaluation process. For every run, I stored the development probabilities, positive-class metrics, selected threshold, model configuration and exported predictions.

## Threshold optimisation

The standard classification threshold of `0.5` was not well suited to this problem.

For each model, I converted its logits into positive-class probabilities and searched for the threshold that maximised positive-class F1 on the development set.

The final selected threshold was `0.08`. This lower threshold allowed the system to recover more subtle positive cases, although it also produced additional false positives. I considered that a reasonable trade-off because the main challenge was detecting a rare and often implicit positive class rather than maximising majority-class accuracy.

## Weighted ensemble

After training the candidate models, I removed runs that fell below a development-quality threshold. I then combined the remaining probability outputs using weights derived from each model's development performance.

The strongest individual model achieved:

```text
Positive-class F1: 0.6359
```

The final four-model weighted ensemble achieved:

```text
Positive-class F1: 0.6413
```

I selected the ensemble only after comparing it against the strongest individual model. This prevented weaker runs from being included simply because additional models were available.

## Error analysis

After producing the final predictions, I manually investigated the false positives and false negatives.

The false positives often mentioned vulnerable groups in supportive or factual contexts. The model recognised language associated with vulnerability but sometimes predicted patronisation even when the surrounding paragraph was neutral.

The false negatives were usually less explicit. These examples could contain patronising framing without the emotional or charity-related vocabulary that appeared frequently in the positive training examples.

Terms repeatedly appearing in the errors included references to homelessness, disability, refugees, poor families, hopelessness and women. This supported one of the concerns I found during the original lexical analysis. Even a contextual transformer could still rely partly on topic and vulnerability cues rather than learning the complete rhetorical distinction.

## Performance across text lengths

I also evaluated positive-class F1 across paragraph-length bands.

| Token length | Positive-class F1 |
|---|---:|
| 0-19 | 0.667 |
| 20-39 | 0.649 |
| 40-79 | 0.637 |
| 80+ | 0.629 |

Performance declined gradually as the paragraphs became longer. Longer examples were generally more contextually complicated, but the model did not experience a sharp collapse in performance.

## Repository structure

```text
.
|-- BestModel/
|   |-- BestModel.ipynb
|   |-- Task5_2_LocalEvaluation.ipynb
|   |-- dev.txt
|   |-- test.txt
|   |-- models_manifest.csv
|   `-- artifacts/
|       `-- final_strategy_summary.csv
|-- EDA/
|   |-- Task2_EDA.ipynb
|   `-- figures/
|-- src/
|   `-- pcl_evaluation/
|       |-- metrics.py
|       |-- thresholding.py
|       `-- ensemble.py
|-- scripts/
|   `-- evaluate_predictions.py
|-- tests/
|   `-- test_evaluation.py
|-- requirements.txt
`-- README.md
```

The notebooks contain the complete exploratory, training and evaluation process. I use the smaller Python modules for the parts of the pipeline that are useful outside the notebooks, particularly metric calculation, threshold selection and probability ensembling.

## Installation

Python 3.10 or later and a CUDA-capable GPU are recommended.

```bash
git clone <repository-url>
cd patronising-language-detection

python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux or macOS:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Data

Place the following files under `data/raw/`:

```text
dontpatronizeme_pcl.tsv
train_semeval_parids-labels.csv
dev_semeval_parids-labels.csv
task4_test.tsv
dontpatronizeme_categories.tsv
```

The original dataset is available through the [Don't Patronize Me! repository](https://github.com/Perez-AlmendrosC/dontpatronizeme).

## Running the project

### Exploratory analysis

Open and run:

```text
EDA/Task2_EDA.ipynb
```

This notebook reproduces the class-distribution, token-length and discriminative n-gram analysis.

### Training and model selection

Open and run:

```text
BestModel/BestModel.ipynb
```

This notebook runs the baseline, RoBERTa-large seed experiments, threshold optimisation, ensemble selection and final prediction export.

### Local evaluation

```bash
python scripts/evaluate_predictions.py \
    --labels path/to/dev_labels.txt \
    --predictions BestModel/dev.txt
```

For CSV labels, pass the relevant column name with `--label-column`. The evaluator reports positive-class precision, recall and F1 together with the complete confusion-matrix counts.

## Main technologies

- Python
- PyTorch
- Hugging Face Transformers and Datasets
- RoBERTa
- scikit-learn
- pandas and NumPy
- Matplotlib and Seaborn

## References

- [SemEval-2022 Task 4: Patronising and Condescending Language Detection](https://aclanthology.org/2022.semeval-1.38/)
- [Don't Patronize Me! An Annotated Dataset with Patronizing and Condescending Language towards Vulnerable Communities](https://aclanthology.org/2020.coling-main.518/)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
