# Datasheet for the fine-tuning corpus

Following Gebru et al. (2021). Documents the assembled fine-tuning corpus
in `data/finetuning/train.jsonl`.

---

## 1. Motivation

**Purpose.** Why was this corpus created? What task does it support?


**Creators.** Who created the corpus and on behalf of which entity?


**Funding.** Who funded the creation of the corpus?


---

## 2. Composition

**Content.** What does each instance in the corpus consist of?


**Count.** How many instances are there in total?


**Breakdown by source.**

| Source | Type | N examples | Proportion |
|--------|------|------------|------------|
| survey_01 | Survey | | |
| survey_02 | Survey | | |
| rct_01 (training split) | RCT | | |
| rct_02 (training split) | RCT | | |
| rct_03 (training split) | RCT | | |
| rct_04 (training split) | RCT | | |
| **Total** | | | |

**Demographic composition.** What is the demographic makeup of the
participants across all component datasets?


**Label distribution.** What is the distribution of outcome labels?


**Missing data.** Are there missing values? How are they handled?


**Confidentiality.** Does the corpus contain data that might be considered
confidential?


---

## 3. Collection process

**How were the component datasets originally collected?**

For each source, summarise the collection method and point to the
corresponding provenance file in `data/human/`.

| Source | Collection method | Provenance file |
|--------|-------------------|-----------------|
| survey_01 | | `data/human/surveys/survey_01/survey_01_provenance.md` |
| survey_02 | | `data/human/surveys/survey_02/survey_02_provenance.md` |
| rct_01 | | `data/human/rcts/rct_01/rct_01_provenance.md` |
| rct_02 | | `data/human/rcts/rct_02/rct_02_provenance.md` |
| rct_03 | | `data/human/rcts/rct_03/rct_03_provenance.md` |
| rct_04 | | `data/human/rcts/rct_04/rct_04_provenance.md` |

**Time period.** Over what time period was the data collected?


**Ethical review.** Were the original studies subject to ethical review?
See `docs/ethics.md` for IRB details.

---

## 4. Preprocessing

**Cleaning.** How were the raw data cleaned?
Implemented in `src/data/cleaning.py`.


**Prompt construction.** How were training prompts constructed from the
cleaned data? Implemented in `src/data/formatting.py`.
Templates are in `data/prompts/`.


**Assembly.** How were the component sources combined into a single
fine-tuning corpus? Implemented in `src/build_corpus.py`.


**Exclusion criteria.** Were any observations excluded? On what basis?


**Transformations.** Were any variables transformed, recoded, or derived?


---

## 5. Uses

**Intended use.** LoRA fine-tuning of large language models for replicating
human behaviour in randomised controlled trials.

**Not intended for:**
- Deployment in production systems
- Clinical or policy decision-making
- Replacing human participants where ethical oversight is required
- Training models for purposes unrelated to RCT replication

**Other known uses.** Has this corpus been used for any other tasks?


---

## 6. Distribution

**Licence.** Under what licence is the corpus distributed?


**Access conditions.** Which component datasets are freely redistributable
and which require data access agreements?

| Source | Redistributable | Access conditions |
|--------|-----------------|-------------------|
| survey_01 | | |
| survey_02 | | |
| rct_01 | | |
| rct_02 | | |
| rct_03 | | |
| rct_04 | | |

**Distribution format.** JSONL (`data/finetuning/train.jsonl`).

---

## 7. Maintenance

**Contact.** Who should be contacted for questions about the corpus?


**Versioning.** How are updates to the corpus tracked?


**Errata.** Is there a process for documenting and communicating errors?

