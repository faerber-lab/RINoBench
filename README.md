# Is this Idea Novel? An Automated Benchmark for Judgment of Research Ideas


> This repository accompanies the paper *Is this Idea Novel? An Automated Benchmark for Judgment of Research Ideas*.  
> It presents a new evaluation benchmark including a dataset of 1,381 research ideas derived from and judged by human experts as well as nine automated evaluation metrics designed to assess both rubric-based novelty scores and textual justifications of novelty judgments.
---

## 🌍 Overview

![example_novelty_judgment.png](figures/example_novelty_judgment.png)

Judging the novelty of research ideas is crucial for advancing science, enabling the identification of unexplored directions, and ensuring contributions meaningfully extend existing knowledge rather than reiterate minor variations. However, given the exponential growth of scientific literature, manually judging the novelty of research ideas through literature reviews is labor-intensive, subjective, and infeasible at scale. Therefore, recent efforts have proposed automated approaches for research idea novelty judgment. Yet, evaluation of these approaches remains largely inconsistent and is typically based on non-standardized human evaluations, hindering large-scale, comparable evaluations. To address this, we introduce **RINoBench**, the first comprehensive benchmark for large-scale evaluation of research idea novelty judgments.
Our benchmark unifies approaches for judging the novelty of research ideas by formalizing the task, illustrated in Figure the Figure above, as the process of comparing a proposed idea with existing work to identify meaningful differences. Further, the task requires predicting a rubric-based novelty score (1–5) alongside a textual justification that grounds the judgment in related literature. This task design enables fine-grained, interpretable judgments of novelty and provides actionable feedback, empowering researchers to iteratively refine their ideas towards greater innovation and impact.

---


## 📂 Repository Structure

```
NoveltyBench/
├── data/
│   ├── final_benchmark_dataset                # includes the dataset of RINoBench
│   ├── evaluations                            # incudes the evaluation results of various state-of-the-art LLMs on RINoBench
│   └── novelty_predictions                    # incudes the reserach idea nvoelty judgments of various state-of-the-art LLMs on RINoBench
│
├── figures/                                   # includes the figures in the paper
│
├── src/
│   ├── data_processing                        # Scripts and LLM prompts used to construct our dataset
│   └── experiments                            # Scripts and LLM prompts used to generate LLM predictions as well as for evaluating the predictions
│
├── .gitignore
├── requirements.txt   
└── README.md
```

## 🧱 Data Description

| File                            | Description                     |
|---------------------------------|---------------------------------|
| `label_descriptions.json`       | The novelty judgment rubric.    |
| `train.json`                    | The train split of our dataset. |
| `test.json`                     | The test split of our dataset.  |

Each train and test split contains research ideas, gold novelty scores, gold textual judgment justifications, related works, and the respective sources from OpenReview.
