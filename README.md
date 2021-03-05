# Big-Scale Analytics 2021 - Team Google

- Francis Ruckstuhl (16-821-738)
- David Netter (16-828-220)
- Loïc Rouiller-Monay (16-832-453)

## Table of Contents

1. Introduction
2. Approach to solve the problem
3. Contribution
4. Bibliography

## 1. Introduction

### The premise of the project

You have decided to form a startup called “LingoRank” with two of your University friends and
become a millionaire. You have until June to create a proof of concept for your investors. Your
startup will revolutionize the way people learn and get better at a foreign language.

You have noticed that, to improve one’s skills in a new foreign language, it is important to read
texts in that language. These text have to be at the reader’s language level. However, it is difficult
to find texts that are close to someone’s knowledge level (A1 to C2). You have decided to build a
model for English speakers that predicts the difficulty of a French written text. This can be then
used, e.g., in a recommendation system, to recommend texts (for example, recent news articles)
that are appropriate for someone’s language level. If someone is at A1 French level, it is
inappropriate to present a text at B2 level, as she won’t be able to understand it. Ideally, a text
should have many known words and may have a few words that are unknown so that the person
can improve.

The project is conducted in multiple milestones that are due throughout the semester.

### Milestone 1 - Reading/Thinking & Gathering the data

The first milestone is to research the subject by reading scientific publications in order to be able to think about how to solve the problem. We ask ourselves questions such as:

- Will we solve the problem as a classification problem, a regression problem, or in another way?
- Will the model analyze the difficulty of each word separately or a sentence as a whole?
- How will the data be collected and labeled in order to train the model?

Secondly, we apply our data collection strategy.

## 2. Approach to solve the problem

At the project's level as a whole, Figure 1 represents the stages we will go through until the final deliverables. Regarding how we will solve the problem of creating a predictive model, we will practice a method inspired by what we have read and learned in the research findings mentioned at the end of the readme. The method is described in the following paragraphs.

First of all, we collected data from reading comprehension tests from examples of exams made available according to each French level by the ex. This allowed us to collect a whole set of more than 1000 sentences sorted by experts from A1 level to C2. Therefore, we guarantee the quality and the accuracy of the training data we provide to the model.

Following this, we will create an NLP Pipeline to transform our sentences into the most efficient features possible to train a machine learning model. To boost accuracy, we will have to extract additional features such as measuring the number of frequencies of words from 1-4 syllables, the diversity of vocabulary per sentence, the number of words, and more on this subject can be found highlighted in yellow in the research findings Microsoft Word file which can be found in the "documents" folder of this repository.

We will treat this problem as a classification problem. However, we have the intuition that it would be possible to treat it as a regression problem with a difficulty that could range from 1 to 6, such as from A1 to C2, while imagining that it would be possible to create a model that can then evaluate a slight difference in level between two texts of the same level as long as a result is higher. For example, two texts with a model that predicts difficulties of 5.16 and 5.38 will be texts judged as C1, but the second would be slightly higher and could offer a more precise system of recommendations to people. However, we will not choose this method. Or, at least, not at this moment.

Regarding the creation of the model, we will use the resources made available on the Google Cloud. We will first try to work with Google AutoML; then, if we feel that we are not satisfied with the solution, it would be possible to do things a little more manually by creating an instance of AI Platform Notebooks (JupyterLab) is a virtual machine instance on which comes preinstalled with the latest machine learning and data science libraries. If that is the case, we will evaluate algorithms that performed well in the papers we read, such as those highlighted in blue in the research findings Microsoft Word file in the "documents" folder of this repository. Some of which are Logistic Regression, AdaBoost, LDA, SVM, kNN, and Neural Networks.

Then we will have to optimize the model, save the best performing model and deploy it. Those stages will be described in more detail later in the course of the project.

```html
<div style="width: 960px; height: 720px; margin: 10px; position: relative;">
  <iframe
    allowfullscreen
    frameborder="0"
    style="width:960px; height:720px"
    src="https://lucid.app/documents/embeddedchart/4ce4bee6-d8cd-49da-9426-4cbe0f52818a"
    id="lgyhY4dA9bO9"
  ></iframe>
</div>
```

## 3. Contribution

Team Google annotated VWXYZ sentences for Milestone 1.

## 4. Bibliography

### Papers

- Balyan, R., McCarthy, K. S., & McNamara, D. S. (2018). Comparing Machine Learning Classification Approaches for Predicting Expository Text Difficulty. In Grantee Submission. https://eric.ed.gov/?id=ED585216
- Balyan, R., McCarthy, K. S., & McNamara, D. S. (2020). Applying Natural Language Processing and Hierarchical Machine Learning Approaches to Text Difficulty Classification. International Journal of Artificial Intelligence in Education, 30(3), 337‑370. https://doi.org/10.1007/s40593-020-00201-7
- Collins, E., Rozanov, N., & Zhang, B. (2018). Evolutionary Data Measures : Understanding the Difficulty of Text Classification Tasks. arXiv:1811.01910 [cs]. http://arxiv.org/abs/1811.01910
- Collins-Thompson, K., & Callan, J. (2004). A Language Modeling Approach to Predicting Reading Difficulty. 193‑200.
- Curto, P., Mamede, N., & Baptista, J. (2015). Automatic Text Difficulty Classifier—Assisting the Selection Of Adequate Reading Materials For European Portuguese Teaching. 36‑44. https://doi.org/10.5220/0005428300360044
- Miltsakaki, E., & Troutt, A. (2008). Real Time Web Text Classification and Analysis of Reading Difficulty. Proceedings of the Third Workshop on Innovative Use of NLP for Building Educational Applications, 89‑97. https://www.aclweb.org/anthology/W08-0911
- Santucci, V., Santarelli, F., Forti, L., & Spina, S. (2020). Automatic Classification of Text Complexity. Applied Sciences, 10(20), 7285. https://doi.org/10.3390/app10207285
- Uchida, S., Takada, S., & Arase, Y. (2018, mai). CEFR-based Lexical Simplification Dataset. Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018). LREC 2018, Miyazaki, Japan. https://www.aclweb.org/anthology/L18-1514
- Wilkens, R., Zilio, L., & Fairon, C. (2018, mai). SW4ALL : A CEFR Classified and Aligned Corpus for Language Learning. Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018). LREC 2018, Miyazaki, Japan. https://www.aclweb.org/anthology/L18-1055

### Source of text data

- Niveaux—Delfdalf.ch. (2021). https://www.delfdalf.ch/niveaux
