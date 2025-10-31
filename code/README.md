# Code

This folder hosts the notebooks and code (in Pyhton) used in the different tutorials and hands-on sessions. The proposed [set-ups](#setup), and [contents](#content) of the sessions are described below.


- [Week 1](#code_week1): Familiarization with BERT-like models, using `transformers` package. Generation of embedding vectors and visualization; applications on word sense disambiguation, and semantic shifts exploration.
- [Week 2](#code_week2)
- [Week 3](#code_week3)
- [Week 4](#code_week4)

## Setups <a name="setup"></a>

Feel free to use the notebooks, either locally, or using hosted services such as Jupyter Binder and Google Colab.

### Running on your machine

You can use the `requirements.txt` file provided at the root of this repository. In your virtual environment, `cd` to repo root, and run:

```bash
pip install -r requirements.txt
```

### Binder

You can launch the projects on Binder: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/d-noe/NLP_DH_PSL_Fall2025/HEAD)

> [!WARNING]
> It can take some time to build the image on Binder.

Binder can be handy to have the repository in a jupyter-lab hosted environment. However, it does not provide extensive memory nor computational resources. Thus, it is not fitted to manipulate large data or to use pre-trained language models with large number of parameters.

### Colab

The notebooks are provided in Google Colab. It provides a convenient way to run the experiments and offers computational resources that should be sufficient for the content of this course in the free-tier (including GPU and TPU runtime access).



## Content <a name="content"></a>

### Week 1 — 29.10 <a name="code_week1"></a>

- [Discover_BERT.ipynb](./1_bert_training/Discover_BERT.ipynb): Familiarize with BERT-like models. Overview of the architecture. Visulisation of attention mechanism. 
- [Tutorial_1_WSD.ipynb](./1_bert_training/Tutorial_1_WSD.ipynb): Familiarize with BERT-like models. Generation of embedding vectors and visualisation. Exemplified with Word Sense Disambiguation application.
- [Hands-on_1_SS.ipynb](./1_bert_training/Hands_on_1_SS.ipynb): Reproduce and expand tutorial's content. Explore Semantic Shifts from LM's lense based on historical newspaper data from [Living With Machines](https://livingwithmachines.ac.uk) initiative.

**Main libraries**: `transformers`, `bertviz`, (`scikit-learn`, `pandas`, `altair`)

<a name="code_supp_1"></a>
<details><summary>To go further</summary> 

- Implement the *attention mechanism* from the [*Attention is All You Need* (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) cornerstone paper in a [Colab Notebook](https://colab.research.google.com/drive/1tm0_Usqkavr0h1Jk0f-ukcykI78xmcfW#scrollTo=tSWEk4ttUgQH) by Alexander "Sasha" Rush. Or read their detailed walkthrough post: [*The Annotated Transformer*](http://nlp.seas.harvard.edu/annotated-transformer/).

</details>

### Week 2 — 05.11 <a name="code_week2"></a>

- [Tutorial_2_MyBERTopic.ipynb](./2_topic_modeling/Tutorial_2_MyBERTopic.ipynb): Implement your own (simplified) version of BERTopic and explore a corpus of 19th century recipes.
- [Hands_on_2_CompareTM.ipynb](./2_topic_modeling/Hands_on_2_CompareTM.ipynb): Apply different topic modeling algorithms on a corpus of UN General Debate speeches from 1946 until today. Try to find the best / most useful representations!

**Main libraries**: `sentence_transformers`, `BERTopic`, `umap`, `hdbscan`, `sklearn`

<a name="code_supp_2"></a>
<details><summary>To go further</summary> 

- [Tutorial - Topic Modeling with BERTopic](https://colab.research.google.com/drive/1FieRA9fLdkQEGDIMYl0I3MCjSUKVF8C-#scrollTo=AXHLDxJdRzBi): A tutorial and overview of the different functionalities of `BERTopic` (Author unknown?).

</details>

### Week 3 — 12.11 <a name="code_week3"></a>

TBD

### Week 4 — 19.11 <a name="code_week4"></a>

TBD