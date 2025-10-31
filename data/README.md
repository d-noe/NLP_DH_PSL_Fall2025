# Data 

This folder hosts the data used in the different tutorials and hands-on sessions. Short descriptions are provided below, please refer to the original sources for further information.

- [Week 1](#data_week1)
  - Word Sense Disambiguation: [CoarseWSD-20](#wsd)
  - Semantic Shifts Analysis: [Living With Machines](#lwm)
- [Week 2](#data_week2)
  - Topic Modeling:
    - [19th century recipes](#19threcipes)
    - [4Newsgroups](#20nwsgrps)
    - [UNGDC](#ungdc)
- [Week 3](#data_week3)
- [Week 4](#data_week4)

The [`preprocessing`](./preprocessing/) folder contains pre-processing scripts used to obtain the presented data files. It is included for transparency and to allow reproduction, but was not developed for pedagogical purposes (-> the code is likely quick and dirty).

## Week 1 — 29.10 <a name="data_week1"></a>

### Tutorial 1: Word Sense Disambiguation <a name="wsd"></a>

- File(s): [`word_sense/CoarseWSD-20.csv`](./word_sense/CoarseWSD-20.csv)
- Description:
  - Dataset based on Wikipedia articles excerpts, developed to evaluate and train word disambiguation models based on a selection of 20 expert-selected words (apple, club, bow, bank, crane, square, chair, java, bass, seal, pitcher, arm, mole, pound, deck, trunk, spring, hood, yard, digit).
  - columns: `text`, `target`, `label`, `label_sense`, `split`
- Source: *Daniel Loureiro, Kiamehr Rezaee, Mohammad Taher Pilehvar, & Jose Camacho-Collados. (2021). Analysis and Evaluation of Language Models for Word Sense Disambiguation.* [![arXiv](https://img.shields.io/badge/arXiv-2008.11608-b31b1b.svg)](https://arxiv.org/abs/2008.11608)

### Hands-on 1: Semantic Shifts <a name="lwm"></a>

Data from the [Living With Machines](https://livingwithmachines.ac.uk) programme. Please check their website if interested in *digital history*, and to discover their research on rethinking the impact of technology on people's lives during the Industrial Revolution.

- File(s): [`word_sense/lwm_annot.csv`](./word_sense/lwm_annot.csv)
- Description:
  - Excerpts of british historical newspaper articles (1818-1920), OCR texts, annotated for word sense for four target words (trolley, car, bike, coach).
  - columns: `text`, `annot`, `newspaper_date`, `newspaper_title`, `newspaper_place`, `image_url`, `image_url_conflict_values`, `target`, `year`.
- Source: Ridge, M., Pedrazzini, N., Vieira, J. M. M., Ciula, A., & McGillivray, B. (2024). Language of mechanisation crowdsourcing datasets from the living with machines project. Journal of Open Humanities Data, 10, 33. [Link](https://kclpure.kcl.ac.uk/portal/en/publications/language-of-mechanisation-crowdsourcing-datasets-from-the-living-) ; [Data Repository](https://bl.iro.bl.uk/concern/datasets/1a677294-cbd3-4bc0-b714-d3bbfd2a6da1)

## Week 2 — 05.11 <a name="data_week2"></a>

### Tutorial 2: Topic Modeling 

#### Nineteenth century American Recipes <a name="19threcipes"></a> 

- File(s): [`topic_data/nineteenth_recipes.csv`](./topic_data/nineteenth_recipes.csv)
- Description:
  - A collection of 1,034 nineteenth-century American recipes, extracted from cookbooks available through the [Project Gutenberg](https://www.gutenberg.org). Each entry includes the recipe text and its corresponding Project Gutenberg book ID.
  - columns: `text`, `ids` (Project Gutenberg ID of the source book)
- Source: Blankenship, A. (2021). A Dataset of Nineteenth-Century American Recipes. Viral Texts: Mapping Networks of Reprinting in 19th-Century Newspapers and Magazines. [GitHub Repository](https://github.com/ViralTexts/nineteenth-century-recipes/)

#### 4Newsgroups <a name="20nwsgrps"></a>

- File(s): [`topic_data/4newsgroups.csv`](./topic_data/4newsgroups.csv)
- Description:
  - The 20 Newsgroups data set is a collection of approximately 18,000 newsgroups posts, partitioned (nearly) evenly across 20 different newsgroups. Here, a filtered version is provided: it is reduced to only 4 topics, and 1,969 documents.
  - columns: `text`, `label` (original source / topic)
- Source: Lang, K. (1995). NewsWeeder: learning to filter netnews. In Proceedings of the Twelfth International Conference on International Conference on Machine Learning (pp. 331–339). Morgan Kaufmann Publishers Inc. [Link](https://dl.acm.org/doi/10.5555/3091622.3091662) ; [Data loaded from scikit-learn handle](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html).

### Hands-on 2: <a name="ungdc"></a>

- File(s): [`topic_data/ungdc.csv`](./topic_data/ungdc_coarse_paragraphs.csv), [``](./topic_data/ungdc_coarse_paragraphs.csv)
- Description:
  - The UN General Debate Corpus (UNGDC) contains 10,952 speeches delivered by representatives from 202 countries (including historical states) during the annual UN General Assembly General Debate from 1946 to 2024. Each speech captures national perspectives on global political issues and provides a longitudinal dataset for the study of international relations, political discourse, and global affairs. An alternative version with coarse heuristic-based paragraphs split is provided as well.
  - columns: 
    - `ungdc.csv`                  : `text`, `country_iso`, `year`, `session`
    - `ungdc_coarse_paragraphs.csv`: `text`, `country_iso`, `year`, `session`, `paragraph_id`
- Source: Jankin, S., Baturo, A., & Dasandi, N. (2024). Words to unite nations: The complete UN General Debate Corpus, 1946–present. Journal of Peace Research. DOI: [10.1177/00223433241275335](https://journals.sagepub.com/doi/10.1177/00223433241275335); [Data Repository](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/0TJX8Y).

## Week 3 — 12.11 <a name="data_week3"></a>

TBD

## Week 4 — 19.11 <a name="data_week4"></a>

TBD