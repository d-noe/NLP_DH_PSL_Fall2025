# Introduction to Natural Language Processing (NLP) — DH PSL,  Fall 2025

This repository hosts material for 4x3hours lectures in the context of the *Introduction to Natural Language Processing (NLP)* class from PSL's Master of Digital Humanities, Fall 2025.

1. [Week 1 (29/10)](#week1)
2. [Week 2 (05/11)](#week2)
3. [Week 3 (12/11)](#week3)
4. [Week 4 (19/11)](#week4)

The code and notebooks for the tutorials and hands-on sessions are provided in the [code](./code/) folder. The data used for these sessions is described and stored in [data](./data/).

## Week 1 (29/10): *Modeling Language: Towards Contextualized Representations* <a name="week1"></a>

- Slides: [preview `html`](https://rawcdn.githack.com/d-noe/NLP_DH_PSL_Fall2025/a3dbb3f4a2901602e813b1262d424bcf20b0dcfe/slides/lecture_1_self_contained.html), [`pdf`](./slides/lecture_1.pdf)
- Notebook(s): [BERT Discovery](./code/1_bert_training/Discover_BERT.ipynb), [Word Sense Disambiguation](./code/1_bert_training/Tutorial_1_WSD.ipynb), [Semantic Shifts](./code/1_bert_training/Hands_on_1_SS.ipynb)
- Key notions: n-gram, transformers, self-attention, context, masked language model / causal language model

<details><summary>To go further</summary>

- [(J. Alammar, 2018)](https://jalammar.github.io/illustrated-transformer/): *The Illustrated Transformer* blog post by Jay Alammar.
- [(Ghaseminejad Raeini, 2025)](https://www.sciencedirect.com/science/article/pii/S2949719125000445): *The evolution of language models: From N-Grams to LLMs, and beyond*.
- [(Allen & Hospedales, 2019)](https://proceedings.mlr.press/v97/allen19a.html): *Analogies Explained: Towards Understanding Word Embeddings*.

**Want more hands-on?** Check the [*To go further* section in code folder](./code/README.md#code_supp_1).

</details>

## Week 2 (05/11): *Discovering Structure: Semantic Spaces & Unsupervised Modeling* <a name="week2"></a>

- Slides: [preview `html`](https://rawcdn.githack.com/d-noe/NLP_DH_PSL_Fall2025/refs/heads/main/slides/lecture_2_self_contained.html), [`pdf`](./slides/lecture_2.pdf)
- Notebook(s): [Custom BERTopic](./code/2_topic_modeling/Tutorial_2_MyBERTopic.ipynb), [Topic Modeling UN General Debates Speeches](./code/2_topic_modeling/Hands_on_2_CompareTM.ipynb)
- Key notions: document representation, BoW, SentenceTransformer, cosine similarity, topic modeling, BERTopic, LDA

<details><summary>To go further</summary>

**Dimensionality Reduction**:
- [(Coenen & Pierce, 2019)](https://pair-code.github.io/understanding-umap/): *Understanding UMAP*: explanations and visual demonstration of UMAP (compared with t-SNE).

**Topic Modeling**:
- [(Churchill & Singh, 2021)](https://doi.org/10.1145/3507900): *The Evolution of Topic Modeling*.
- [(Li et al., 2024)](https://doi.org/10.1515/dsll-2024-0010): *Applying Topic Modeling to Literary Analysis: A Review*.
- [(Gillings & Hardie, 2022)](https://doi.org/10.1093/llc/fqac075): *The interpretation of topic models for scholarly analysis: An evaluation and critique of current practice*.
- [(Antoniak, 2023)](https://maria-antoniak.github.io/2022/07/27/topic-modeling-for-the-people.html): *Topic Modeling for the People*: an interesting blogpost by Maria Antoniak, sharing a set of steps that you can follow to get coherent topics from most datasets, primarily focusing on LDA. It provides as well many additional references to dig deeper.
- [(Egger & Yu, 2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9120935/): *A Topic Modeling Comparison Between LDA, NMF, Top2Vec, and BERTopic to Demystify Twitter Posts*.
- **Evaluation Concerns**:
    - [(Chang et al., 2009)](https://papers.nips.cc/paper_files/paper/2009/hash/f92586a25bb3145facd64ab20fd554ff-Abstract.html): *Reading Tea Leaves: How Humans Interpret Topic Models*.
    - [(Hoyle et al., 2021)](https://proceedings.neurips.cc/paper/2021/file/0f83556a305d789b1d71815e8ea4f4b0-Paper.pdf): *Is Automated Topic Model Evaluation Broken?: The Incoherence of Coherence*.

**Want more hands-on?** Check the [*To go further* section in code folder](./code/README.md#code_supp_2).

</details>

## Week 3 (12/11): *Inferring Patterns: Supervised Tasks and Adaptation* <a name="week3"></a>

- Slides: [preview `html`](), [`pdf`](./slides/lecture_3.pdf)
- Notebook(s): [BERT Fine-Tuning Tutorial](./code/3_supervised/Tutorial_3_SFT.ipynb), [Canonicity Prediction Challenge: Performance and Fairness](./code/3_supervised/Hands_on_3_CanonChallenge.ipynb)
- Key notions: classification, supervised fine-tuning, performance metrics, fairness

<details><summary>To go further</summary>

**Text Classification for DH**
- [(Bamman et al., 2024)](https://ceur-ws.org/Vol-3834/paper119.pdf): *On Classification with Large Language Models in Cultural Analytics*.
- [(Lassen et al., 2024)](https://ceur-ws.org/Vol-3834/paper76.pdf): *Literary Canonicity and Algorithmic Fairness: The Effect of Author Gender on Classification Models*.

**Fairness & Bias**
- [(Solon Barocas, Moritz Hardt, Arvind Narayanan, 2023)](https://fairmlbook.org): *FAIRNESS AND MACHINE LEARNING - Limitations and Opportunities* — Full book available with additional resources.
- [(Irving & Askell, 2019)](10.23915/distill.00014): *AI Safety Needs Social Scientists*.
- [(Blodgett et al., 2020)](https://aclanthology.org/2020.acl-main.485/): *Language (Technology) is Power: A Critical Survey of ``Bias'' in NLP.*
- [(Hovy & Prabhumoye)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9285808/pdf/LNC3-15-0.pdf): *Five sources of bias in natural language processing*.
- [(Gallegos et al., 2024)](https://aclanthology.org/2024.cl-3.8.pdf): *Bias and Fairness in Large Language Models: A Survey*.

**Interpretability**
- [Interpretability Blog-post](https://blog.ml.cmu.edu/2020/08/31/6-interpretability/)
- [(Olah et al., 2018)](10.23915/distill.00010): *The Building Blocks of Interpretability*: Mainly focusing (or applying) on computer vision, but a very nice and illustrated article on interpretability. 

</details>

## Week 4 (19/11):  *LLMs** (TBD) <a name="week4"></a>

- Slides: 
- Notebook(s): 
- Key notions: LLMs

<details><summary>To go further</summary>

- [(Gallegos et al., 2025)](https://direct.mit.edu/coli/article/50/3/1097/121961/Bias-and-Fairness-in-Large-Language-Models-A): *Bias and Fairness in Large Language Models: A Survey*

</details>
