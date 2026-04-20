# REFORMS checklist

**Source:** Kapoor et al. (2024). REFORMS: Consensus-based Recommendations
for Machine-learning-based Science. *Science Advances*, 10, eadk3452.
https://www.science.org/doi/10.1126/sciadv.adk3452

32 items across 8 modules. For each item, mark the response and provide
the file path or paper section where the requirement is addressed.

---

## Module 1: Study goals (3 items)

**1a.** State the research question and its significance.
- [ ] Yes  [ ] No  [ ] N/A
- Location:
- Notes:

**1b.** State the contribution of ML to addressing the research question
(e.g., prediction, causal inference, exploration, description).
- [ ] Yes  [ ] No  [ ] N/A
- Location:
- Notes:

**1c.** Justify the choice of ML over alternative methods.
- [ ] Yes  [ ] No  [ ] N/A
- Location:
- Notes:

---

## Module 2: Computational reproducibility (5 items)

**2a.** Provide access to the data, or if access is restricted, describe
how to obtain the data.
- [ ] Yes  [ ] No  [ ] N/A
- Location:
- Notes:

**2b.** Provide access to the code used for data preprocessing, model
training, model evaluation, and analysis.
- [ ] Yes  [ ] No  [ ] N/A
- Location:
- Notes:

**2c.** Specify the computational requirements (hardware, runtime,
dependencies) needed to reproduce the results.
- [ ] Yes  [ ] No  [ ] N/A
- Location:
- Notes:

**2d.** Provide step-by-step instructions for reproducing the results.
- [ ] Yes  [ ] No  [ ] N/A
- Location:
- Notes:

**2e.** Provide a single entry point (e.g., a master script) that
reproduces all results, figures, and tables from raw data.
- [ ] Yes  [ ] No  [ ] N/A
- Location:
- Notes:

---

## Module 3: Data quality (7 items)

**3a.** Describe the provenance of the data, including how it was
collected and by whom.
- [ ] Yes  [ ] No  [ ] N/A
- Location:
- Notes:

**3b.** Describe the population represented by the data, including
the sampling strategy.
- [ ] Yes  [ ] No  [ ] N/A
- Location:
- Notes:

**3c.** Report the size of the dataset (number of instances, number
of features).
- [ ] Yes  [ ] No  [ ] N/A
- Location:
- Notes:

**3d.** Describe the data type of each feature (e.g., continuous,
categorical, ordinal, text, image).
- [ ] Yes  [ ] No  [ ] N/A
- Location:
- Notes:

**3e.** Describe how missing data was handled.
- [ ] Yes  [ ] No  [ ] N/A
- Location:
- Notes:

**3f.** Describe any data quality issues encountered and how they
were addressed.
- [ ] Yes  [ ] No  [ ] N/A
- Location:
- Notes:

**3g.** If using an existing dataset, cite the original source and
describe any modifications made.
- [ ] Yes  [ ] No  [ ] N/A
- Location:
- Notes:

---

## Module 4: Data preprocessing (3 items)

**4a.** Describe all data transformations applied (e.g., normalization,
encoding, imputation, feature extraction).
- [ ] Yes  [ ] No  [ ] N/A
- Location:
- Notes:

**4b.** Document any observations excluded from the analysis and
the criteria for exclusion.
- [ ] Yes  [ ] No  [ ] N/A
- Location:
- Notes:

**4c.** If creating derived variables or features, describe how they
were constructed and justify their inclusion.
- [ ] Yes  [ ] No  [ ] N/A
- Location:
- Notes:

---

## Module 5: Modeling (6 items)

**5a.** Specify the ML model(s) used, including:
  - model architecture or algorithm
  - number of parameters
  - key implementation details
- [ ] Yes  [ ] No  [ ] N/A
- Location:
- Notes:

**5b.** Justify the choice of model(s) for the task.
- [ ] Yes  [ ] No  [ ] N/A
- Location:
- Notes:

**5c.** Report all hyperparameters and how they were selected
(e.g., grid search, random search, manual tuning, default values).
- [ ] Yes  [ ] No  [ ] N/A
- Location:
- Notes:

**5d.** Describe the training procedure, including the loss function,
optimizer, learning rate schedule, and stopping criteria.
- [ ] Yes  [ ] No  [ ] N/A
- Location:
- Notes:

**5e.** Describe the data splitting strategy, including:
  - how data was divided into training, validation, and test sets
  - the rationale for the splitting method
  - whether any stratification was applied
- [ ] Yes  [ ] No  [ ] N/A
- Location:
- Notes:

**5f.** Report the random seeds used and the number of repetitions.
- [ ] Yes  [ ] No  [ ] N/A
- Location:
- Notes:

---

## Module 6: Data leakage (3 items)

**6a.** Confirm that the test set was not used during model development
(training or hyperparameter selection).
- [ ] Yes  [ ] No  [ ] N/A
- Location:
- Notes:

**6b.** If observations are not independent (e.g., repeated measures,
spatial or temporal correlation, group structure), describe how
data splitting accounted for this.
- [ ] Yes  [ ] No  [ ] N/A
- Location:
- Notes:

**6c.** Confirm that no information from the target variable leaked
into the features (e.g., through preprocessing steps applied
before splitting).
- [ ] Yes  [ ] No  [ ] N/A
- Location:
- Notes:

---

## Module 7: Metrics and uncertainty (3 items)

**7a.** Report the evaluation metrics used and justify their choice
for the research question.
- [ ] Yes  [ ] No  [ ] N/A
- Location:
- Notes:

**7b.** Report uncertainty estimates for all results (e.g., confidence
intervals, standard errors, variance across seeds or folds).
- [ ] Yes  [ ] No  [ ] N/A
- Location:
- Notes:

**7c.** If performing statistical tests, report the test used, the
test statistic, and the p-value.
- [ ] Yes  [ ] No  [ ] N/A
- Location:
- Notes:

---

## Module 8: Interpretation and generalizability (2 items)

**8a.** Discuss the limitations of the study, including threats to
validity and generalizability.
- [ ] Yes  [ ] No  [ ] N/A
- Location:
- Notes:

**8b.** Discuss the broader implications of the findings and any
ethical considerations.
- [ ] Yes  [ ] No  [ ] N/A
- Location:
- Notes:
