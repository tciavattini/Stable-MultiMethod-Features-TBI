# Machine Learning-driven biomarker discovery for stratifying treatment response in tick-borne illness

**Teresa Ciavattini, Marc Shawky, Séverine Padiolleau-Lefèvre, Florian De Vuyst, Gordana Avramovic, John Shearer Lambert**

> Ciavattini T et al. *Machine Learning-driven biomarker discovery for stratifying treatment response in tick-borne illness.* Manuscript under submission.

---

## Abstract

Lyme disease and related tick-borne infections can cause persistent symptoms even after antibiotic treatment. Currently, clinicians cannot reliably predict which patients will respond well to therapy. We analysed clinical and laboratory data from 301 patients collected before treatment. By systematically evaluating 149 patient characteristics across 100 repeated analyses using multiple methods (over 700,000 evaluations), we identified 22 features that consistently distinguished responders from non-responders. Patients who responded well tended to report greater physical symptom burden at baseline, whereas non-responders reported better mood and overall well-being. Five features were consistently identified across all analytical approaches. These findings suggest that baseline symptom and immune profiles may help identify patients more likely to benefit from treatment and support more informed clinical decision-making.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Pipeline Overview](#pipeline-overview)
3. [Data](#data)
4. [Repository Structure](#repository-structure)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Results](#results)
8. [Key Contributions](#key-contributions)
9. [Reproducibility](#reproducibility)
10. [Licenses](#licenses)
11. [Feedback](#feedback)
12. [References](#references)
13. [Citation](#citation)

---

## Introduction

Lyme disease and associated tick-borne co-infections represent a growing public health burden, with an estimated 500,000 new cases per year in the United States and comparable incidence across endemic European regions. While antibiotic therapy is effective for most patients, 10% to 36% report persistent or recurring symptoms following treatment, including severe fatigue, musculoskeletal pain, neurological dysfunction, and impaired mood, a condition formally classified as post-treatment Lyme disease syndrome (PTLDS).

The inability to identify at baseline which patients are likely to respond to treatment remains an unsolved clinical problem. Patients unlikely to respond might benefit from earlier escalation; likely responders could be spared unnecessary intensification. Machine learning offers advantages over conventional univariate or stepwise approaches for identifying such discriminators, as it can evaluate multiple candidate features simultaneously while accounting for their interactions.

However, a critical methodological barrier limits clinical translation: in small datasets where candidate features approach or exceed the number of patients, feature selection results are highly sensitive to the choice of algorithm and data partitioning, often producing findings that fail to replicate. Standard sparsity-promoting methods frequently yield unstable feature sets across independent cohorts.

This repository implements a **stability-aware machine learning framework** that addresses these limitations by aggregating evidence across 100 independent resampling iterations and multiple selection methods, formally quantifying reproducibility using the Nogueira stability index, Jaccard similarity, and Spearman rank concordance, and retaining only features that are consistently discriminative regardless of algorithmic choice or sampling variation.

The full analytical pipeline is publicly available to support independent replication, in alignment with TRIPOD+AI reporting principles.

---

## Pipeline Overview

![Pipeline overview](figures/main/pipeline_workflow.jpg)

---

## Data

The data folder contains a CSV file with the following columns:

**Demographics and disease history**
| Column | Description |
|--------|-------------|
| `age` | Patient age at baseline |
| `gender` | Patient sex |
| `persistent` | Persistent symptom status |
| `Q1_Residence_Ireland` | Residence in Ireland |
| `Q2_residence_outside_Ireland` | Residence outside Ireland |
| `Q3_Outdoor_hobbies` | Outdoor hobbies (tick exposure risk) |
| `Q4_Chronic_previous` | Previous chronic conditions |
| `Q5_tick_bite` | History of tick bite |
| `Q6_tick_bite_when` | Time of tick bite |
| `Q8_Interval` | Interval since tick bite |

**Healthcare access and satisfaction**
| Column | Description |
|--------|-------------|
| `Q9_GP` | GP consultations |
| `Q9a_GP_satisfaction` | GP satisfaction rating |
| `Q10_Consultant` | Specialist consultations |
| `Q10a_consultant_satisfaction` | Consultant satisfaction rating |
| `Q11_Number_doc` | Number of doctors consulted |
| `Q12_trt_care_rate` | Treatment and care rating |

**Employment status**
| Column | Description |
|--------|-------------|
| `Working` | Currently working |
| `Sick Leave` | On sick leave |
| `Retired` | Retired |
| `Caring res` | Caring responsibilities |
| `Unemplo` | Unemployed |
| `Other_empl` | Other employment status |
| `Q14_Impact_symp_employment` | Impact of symptoms on employment |

**Baseline symptom burden (binary presence/absence)**
| Column | Description |
|--------|-------------|
| `Q15 Bulls Eye` | Bull's eye rash |
| `Q16. Rash` | Rash |
| `Q17. Sweats` | Sweats |
| `Q18 Sore throat` | Sore throat |
| `Q19. Headac` | Headache |
| `Q20. Swglands` | Swollen glands |
| `Q21. Sev Fat` | Severe fatigue |
| `Q24. Chest P.` | Chest pain |
| `Q25. ShortB` | Shortness of breath |
| `Q26. Palpit` | Palpitations |
| `Q27. Lighthead.` | Lightheadedness |
| `Q28. JointP` | Joint pain |
| `Q29.Moving` | Pain on moving |
| `Q30. Intensity` | Pain intensity |
| `Q31. JointSw` | Joint swelling |
| `Q32. MuscW.le` | Muscle weakness |
| `Q33.MuscP` | Muscle pain |
| `Q35Facial` | Facial symptoms |
| `Q36ArmHan` | Arm/hand symptoms |
| `Q37Numbn` | Numbness |
| `Q38Concen` | Concentration difficulties |
| `Q39Sleep` | Sleep disturbance |
| `Q40Vision` | Vision problems |
| `Q41. Neck` | Neck stiffness |
| `Q42Tinnit` | Tinnitus |
| `Q43Person` | Personality changes |
| `Q44Mood` | Mood changes |
| `Q46Anger` | Anger |
| `Q47Anxiety` | Anxiety |

**Baseline symptom severity scores (1–10 numeric rating scales)**
| Column | Description |
|--------|-------------|
| `T0_severe_fatigue_rate` | Fatigue severity — burden scale (lower = better) |
| `T0_muscle_pain_rate` | Muscle pain severity — burden scale (lower = better) |
| `T0_symp_today_rate` | Overall symptom severity — well-being scale (higher = better) |
| `T0_mood_rate` | Mood severity — well-being scale (higher = better) |

**Treatment history**
| Column | Description |
|--------|-------------|
| `Q48_blood_analysis` | Blood analysis performed |
| `Q49_blood_analysis_where` | Location of blood analysis |
| `Q50_antibiotic` | Prior antibiotic use |
| `Q52_antib_duration` | Prior antibiotic duration |
| `Q53_antib_symp_improv` | Symptom improvement with prior antibiotics |
| `Q54_alternative_trt` | Alternative treatments used |
| `Q55_alternative_trt_success` | Success of alternative treatments |

**Treatment doses (study regimen)**
| Column | Description |
|--------|-------------|
| `Cefuroxime_dose` | Cefuroxime dose |
| `Rifampicin_dose` | Rifampicin dose |
| `Lymecyclin_dose` | Lymecycline dose |
| `Azithromycin_dose` | Azithromycin dose |
| `Clarithromycin_dose` | Clarithromycin dose |
| `Doxycycline_dose` | Doxycycline dose |
| `Amoxicillin_dose` | Amoxicillin dose |
| `LDN_dose` | Low-dose naltrexone dose |
| `Melatonin_dose` | Melatonin dose |
| `Valoid_dose` | Valoid dose |
| `Malarone_dose` | Malarone dose |
| `Diflucan_dose` | Diflucan dose |

**Serological profile (TICKPLEX® ELISA)**
| Column | Description |
|--------|-------------|
| `b.burg+afz+gar.IgG` / `IgM` | *Borrelia burgdorferi* s.l. IgG/IgM |
| `B.Burg Round Body IgG` / `IgM` | *B. burgdorferi* round body IgG/IgM |
| `BaB M IgG` / `IgM` | *Babesia microti* IgG/IgM |
| `Bart H IgG` / `IgM` | *Bartonella henselae* IgG/IgM |
| `Ehrl C IgG` / `IgM` | *Ehrlichia chaffeensis* IgG/IgM |
| `Rick Ak IgG` / `IgM` | *Rickettsia akari* IgG/IgM |
| `Coxs IgG` / `IgM` | Coxsackievirus IgG/IgM |
| `Epst B IgG` / `IgM` | Epstein–Barr virus IgG/IgM |
| `Hum Par IgG` / `IgM` | Human parvovirus IgG/IgM |
| `Mycop Pneu IgG` / `IgM` | *Mycoplasma pneumoniae* IgG/IgM |
| `Chlamydia pneumoni` | *Chlamydia pneumoniae* |
| `HSV / IgG`, `HSV/1gG` | Herpes simplex virus IgG |
| `VZV/ IgG` | Varicella-zoster virus IgG |
| `Toxoplasma` | *Toxoplasma gondii* |
| `Yersinia`, `Yersinia.1` | *Yersinia* spp. |
| `BB Full Anti`, `BB Osp Mix`, `BBLFA` | *Borrelia* full antigen / OspMix / LFA panels |
| `Babesia`, `Bartonella H`, `Myco Pne`, `Ehrl Ana` | Composite seropositivity flags |
| `Rickettsia`, `Epstein B`, `Chlamydia P`, `Chlamyd trac` | Composite seropositivity flags |
| `Cytomigalo`, `VZV IgG`, `Anaplasma Phago`, `Herpes Simplex` | Composite seropositivity flags |
| `Aspergillas`, `Candida`, `NVRLIgG` | Fungal and reference lab flags |

**Immunological markers**
| Column | Description |
|--------|-------------|
| `CD3%` | T lymphocyte percentage |
| `CD3Total` | Total T lymphocyte count |
| `CD4%` | Helper T cell percentage |
| `CD4-Helper` | CD4+ Helper T cell count |
| `CD8%` | Cytotoxic T cell percentage |
| `CD8-Suppr` | CD8 suppressor count |
| `H/SRATIO` | Helper/suppressor ratio |
| `CD19Bcell` | B cell count |
| `CD19%` | B cell percentage |
| `CD57+NKCELLS` | CD57+ NK cell count |
| `IgG` | Immunoglobulin G |
| `IgA` | Immunoglobulin A |
| `IgM` | Immunoglobulin M |

**Routine haematology and biochemistry**
| Column | Description |
|--------|-------------|
| `HgB` | Haemoglobin |
| `Platelets` | Platelet count |
| `neutrophils` | Neutrophil count |
| `Lymphocytes` | Lymphocyte count |
| `WCC` | White cell count |
| `CRP` | C-reactive protein |
| `RF` | Rheumatoid factor |
| `ANA` | Antinuclear antibodies |
| `Iron` | Serum iron |
| `Transf` | Transferrin |
| `%transsat` | Transferrin saturation |
| `Ferritin` | Serum ferritin |
| `Folate` | Serum folate |
| `CK` | Creatine kinase |
| `FT4` | Free thyroxine |
| `TSH` | Thyroid-stimulating hormone |

**Outcome variables (T2 follow-up — not used as baseline features)**
| Column | Description |
|--------|-------------|
| `T2_symp_today_rate` | Overall symptom severity at T2 |
| `T2_muscle_pain_rate` | Muscle pain severity at T2 |
| `T2_severe_fatigue_rate` | Fatigue severity at T2 |
| `T2_mood_rate` | Mood severity at T2 |


---

## Repository Structure

```
.
├── dataset/            # Anonymised dataset (where applicable)
├── src/                # Core pipeline scripts
├── figures/            # Figures used in README and paper
├── requirements.txt    # Python dependencies
├── environment.yml
├── MIT license
├── CC-BY-SA-4.0 license
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/tciavattini/Stable-MultiMethod-Features-TBI.git
cd Stable-MultiMethod-Features-TBI
```

Create and activate a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Python 3.10 is required.

---

## Usage

The analysis pipeline is organised into sequential scripts. Run them in the following order:

**Step 1 — Feature engineering**
```bash
src/01_feature_engineering_pipeline.ipynb
```
Handles raw data ingestion, recoding of categorical variables, aggregation into domain-specific symptom counts, and exclusion of post-baseline measurements. Produces the final 149-variable baseline feature matrix.

**Step 2 — Outcome classification**
```bash
python src/02_outcome_classification.py
```
Derives the composite treatment response score from longitudinal changes across four symptom domains (severe fatigue, muscle pain, overall symptoms, mood) and stratifies patients into high responders, middle responders, and non-responders by tertiles.

**Step 3 — Data preparation**
```bash
python src/03_prepare_data.py
```
Prepares formatted input arrays for downstream modelling.

**Step 4 — Sanity check**
```bash
python src/sanity_check.py
```
Validates data integrity, class balance, and preprocessing correctness before running the main analysis.

**Step 5 — Stability analysis (main)**
```bash
python src/04_stability_top30.py
```
Core of the pipeline. Runs 100 independent resampling iterations, each comprising 10 independent 80/20 stratified train/test splits. Assesses each of the 149 features across five scoring criteria (Linear SVM, k-NN, neural network, Linear Regression R², information gain), computes consensus rankings, and identifies stable features at ≥70% selection frequency (~700,000 evaluations total). Note: requires `data_loading.py`.

To run in background with logging:
```bash
nohup python src/04_stability_top30.py > logs/stability_run_$(date +%Y%m%d_%H%M).log 2>&1 &
echo $! > logs/stability.pid
```

**Step 6 — Multi-feature consensus**
```bash
python src/05_multifeature.py
```
Aggregates feature rankings across methods into a consensus score. Evaluates classification performance across feature set configurations (full set, stable set, random baselines, cumulative top-k sets) using Linear SVM, k-NN, and Decision Tree under 10×10 repeated stratified cross-validation.

**Step 7 — Alternative feature selection methods**
```bash
python src/06_alt_feature_selection.py
python src/07_alt_fs_classification.py
```
Independently applies three embedded methods (LASSO, Elastic Net, Random Forest) under identical resampling conditions. Computes a three-method combined consensus and identifies the robust core feature set selected by ≥3 of 4 approaches.

**Step 8 — Phenotype analysis**
```bash
python src/08_phenotype_analysis.py
```
Characterises clinical phenotypes associated with stable feature subsets. Computes two-sided Mann–Whitney U tests with Benjamini–Hochberg FDR correction (α=0.05) and effect sizes (Cohen's *d*, Common Language Effect Size) for the 22 stable features.

**Step 9 — Sensitivity analyses**
```bash
python src/09_sensitivity_analyses.py
```
Evaluates robustness of findings under alternative stability thresholds (50–90%), alternative hyperparameter specifications, and alternative modelling assumptions.

All experiments are executed with fixed random seeds to ensure full reproducibility.

---

## Results

### Stable Feature Identification

Across 100 resampling iterations, **22 baseline features** demonstrated stable selection frequency (≥70%), corresponding to 14.8% of the original feature space. The highest-ranked features were predominantly somatic symptom measures: severe fatigue severity, neurological symptom count, swollen glands, muscle pain severity, and mood severity. Immunological markers (CD8+, CD3+ T cells), demographic variables (age, employment status), and treatment-related variables were also retained.

Stability metrics confirmed moderate-to-good reproducibility: Nogueira stability index = 0.70, mean pairwise Jaccard similarity = 0.61, Spearman rank concordance *r* = 0.87 (*p* < 0.001).

![Stable features](figures/main/stable_features.jpg)

### Multi-Feature Classification Performance

Models trained on the 22 stable features consistently outperformed those trained on the full 149-feature set and random subsets of equal size across all three classifiers:

| Feature set | Classifier | Accuracy | AUC |
|---|---|---|---|
| Stable (22) | Linear SVM | 0.641 [0.619, 0.663] | 0.708 [0.681, 0.734] |
| Stable (22) | k-NN | 0.646 [0.620, 0.671] | 0.710 [0.681, 0.737] |
| Stable (22) | Decision Tree | 0.635 [0.612, 0.658] | 0.651 [0.626, 0.675] |
| Full (149) | Linear SVM | 0.529 [0.506, 0.553] | 0.534 [0.507, 0.562] |
| Random (22) | Linear SVM | 0.531 [0.521, 0.541] | 0.539 [0.526, 0.552] |

The stable feature set significantly outperformed both baselines across all classifiers (permutation *p* < 0.001; Cohen's *d* = 0.83–1.14). Classification performance plateaued at approximately five top-ranked features and declined gradually beyond 15.

### Robust Core Features

Five features were selected by all four methodologies (Vendrow consensus, LASSO, Elastic Net, Random Forest): **CD8+ T cells (%), severe fatigue severity, overall symptom severity, muscle pain severity, and mood severity**.

### Phenotypic Characterisation

High responders were characterised by elevated somatic symptom burden (higher fatigue, muscle pain, neurological symptoms) coupled with relatively preserved mood at baseline. Non-responders exhibited the inverse pattern: better mood and self-rated wellbeing but lower somatic burden. Seven of the 22 stable features reached individual significance after FDR correction (*q* < 0.05; effect sizes |*d*| = 0.44–0.69).

---

## Key Contributions

- Stability-aware feature selection framework for clinical ML in heterogeneous, small-sample cohorts
- Multi-method consensus ranking that reduces dependence on any single algorithm or sampling choice
- Formal quantification of feature reproducibility using Nogueira stability index, Jaccard similarity, and Spearman rank concordance
- Demonstration that apparent performance advantages can arise from shared regularisation assumptions between selection and classification steps, rather than genuine generalisation
- Fully reproducible pipeline aligned with TRIPOD+AI reporting principles

---

## Reproducibility

- Python 3.10.13
- scikit-learn 1.5.1 · NumPy 1.26.4 · pandas 2.2.2 · SciPy 1.14.1
- Fixed random seeds for all stochastic processes
- Deterministic pipeline execution order
- Full environment specification provided via `requirements.txt`

---

## Licenses

- **Code**: MIT License — see `MIT license` file for details
- **Dataset**: Creative Commons BY-SA 4.0 — see `CC-BY-SA-4.0 license` file for details

---

## Feedback

We welcome questions, suggestions, and contributions. Please open an issue on GitHub if you encounter any problems with the repository or would like to discuss the methodology.

---

## References

1. Xi, D. et al. A Longitudinal Study of a Large Clinical Cohort of Patients with Lyme Disease and Tick-Borne Co-Infections Treated with Combination Antibiotics. *Microorganisms* 2023, 11, 2152. https://doi.org/10.3390/microorganisms11092152

2. Xi, D. et al. Scrutinizing Clinical Biomarkers in a Large Cohort of Patients with Lyme Disease and Other Tick-Borne Infections. *Microorganisms* 2024, 12, 380. https://doi.org/10.3390/microorganisms12020380

3. Garg, K. et al. Biomarker-Based Analysis of Pain in Patients with Tick-Borne Infections before and after Antibiotic Treatment. *Antibiotics* 2024, 13, 693. https://doi.org/10.3390/antibiotics13080693

4. Vendrow, J. et al. Feature Selection from Lyme Disease Patient Survey Using Machine Learning. *Algorithms* 2020, 13, 334. https://doi.org/10.3390/a13120334

5. Nogueira, S., Sechidis, K. & Brown, G. On the stability of feature selection algorithms. *J. Mach. Learn. Res.* 2017, 18, 6345–6398.

---

## Citation

If you use this repository, please cite the associated manuscript:

> Ciavattini T, Shawky M, Padiolleau-Lefèvre S, De Vuyst F, Avramovic G, Lambert JS. *Machine Learning-driven biomarker discovery for stratifying treatment response in tick-borne illness.* Manuscript under submission.

```bibtex
@misc{Stable-MultiMethod-Features-TBI,
  author    = {Teresa Ciavattini and Marc Shawky and S{\'e}verine Padiolleau-Lef{\`e}vre
               and Florian {De Vuyst} and Gordana Avramovic and John Shearer Lambert},
  title     = {Machine Learning-Driven Biomarker Discovery for Stratifying Treatment
               Response in Tick-Borne Illness},
  year      = {2025},
  publisher = {GitHub},
  journal   = {GitHub repository},
  howpublished = {\url{https://github.com/tciavattini/Stable-MultiMethod-Features-TBI}},
}
```

---

**Funding.** This project is co-funded by the European Union's Horizon Europe research and innovation programme Cofund SOUND.AI under the Marie Skłodowska-Curie Grant Agreement No 101081674, by Hauts-de-France Region (STIMulE, STIP, DiaLyme), and by the Institut des Sciences du Calcul et des Données (ISCD) of Sorbonne University via the Num4Lyme project.

---

**Teresa Ciavattini**  
Université de Technologie de Compiègne / Sorbonne Université – SCAI
