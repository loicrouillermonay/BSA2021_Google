# Big-Scale Analytics 2021 - Team Google

- Francis Ruckstuhl
- David Netter
- Loïc Rouiller-Monay

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

### Milestone 2 - Creating/Evaluating the model

The second milestone is to create a predictive model that can predict how easy or difficult a french sentence is. At the end of the milestone, the deliverable is a callable API point which, given a text in French, will return its difficulty. Part of the exercise is to create an API, dockerize the solution and use cloud services.

## 2. Approach to solve the problem

To solve the problem of creating a predictive model, we will use a method inspired from the papers we have read and learned in the research findings mentioned at the end of the readme. The method is described in the following paragraphs.

First of all, we collected data from reading comprehension tests from samples of exams made available by the "Fondation Esprit Francophonie DELF DALF Suisse". Those texts are already labeled according to their difficulty level. This allowed us to collect a whole set of more than 1000 sentences sorted by experts from A1 level to C2. Therefore, the quality and the accuracy of the training data is guaranteed.

Following this, we will create an NLP Pipeline to transform the sentences into the most efficient features possible to train a machine learning model. To boost accuracy, we will have to extract additional features such as measuring the number of frequencies of words from 1-4 syllables per sentence, calculate the diversity of vocabulary per sentence with specific formulas or external python libraries such as "wordstats", count the length of each sentence and use various lexical treatments common to NLP such as lemmatization. More on this subject can be found highlighted in yellow in the "research findings" Microsoft Word file in the "documents" folder of this repository.

Thus, it has been decided that we will treat this assignment as a classification problem and we will take into consideration each sentence as a whole for the difficulty classification. However, we have the intuition that it would be possible to treat it as a regression problem; with a difficulty that could range from 1 to 6, such as from A1 to C2, it would be possible to create a model that can then evaluate a slight difference in level between two texts of the same level as long as a result is higher. For example, two texts with a model that predicts difficulties of 5.16 and 5.38 will be texts judged as C1, but the second would be slightly higher and could offer a more precise system of recommendations to people. However, we will not choose this method. Or, at least, not at this time in the project.

Regarding the creation of the model, we will use the resources made available on the Google Cloud. We will first upload the data to the Google Cloud Platform and try to work with Google AutoML; then, if we feel that we are not satisfied with the solution, it would be possible to do things a more manually by creating an instance of AI Platform Notebooks (JupyterLab). It is a virtual machine instance that comes preinstalled with the latest machine learning and data science libraries. If that is the case, we will evaluate algorithms that performed well in the papers we read, such as those highlighted in blue in the research findings Microsoft Word file in the "documents" folder of this repository. Some of those are Logistic Regression, AdaBoost, LDA, SVM, kNN, and Neural Networks.

Then we will have to optimize the model, save the best performing model and deploy it. Those stages will be described in more detail later in the course of the project.

## 3. Contribution

Team Google annotated 1020 sentences for Milestone 1.

## 4. Synthesis of the work done on Milestone 2

Much work has been done on milestone two, and this chapter summarizes it in a precise and short way. This chapter will first describe our strategy regarding the training data, then explain the creation of three types of models that we did in parallel, then finally how we deployed this model to make an API and a graphical interface.

### A quick word on the data

To make it simple, what we will describe in the following chapters, we first did it with our data set that we had collected. However, we had very little data and in the end, it was a matter of knowing which model underfitted the least. That is why we changed our strategy and selected 9174 observations from our data and those of our colleagues that we found the most qualitative. In this sense, we could finally work on models that were not constantly underfitting to know which one will be chosen for the final milestone of this project, where everyone will train with the same dataset.

#### Predictive models

Three types of models were created in parallel to evaluate the results and choose the best one. To put the results in context:

- Base rate: 0.16
- Default Rate: 0.19

All the notebooks for each of the models and the one for data preparation are in the "notebooks" folder in the GitHub archive.

##### A. Google Cloud Platform - Natural Language

This is the simplest solution. All data with a text column and a column with the label were uploaded to Google Storage and then were transmitted to the product "Natural Language". Then Google creates a model by itself via its text classification wizard. The training lasts one day, and an email was sent to us where the model's accuracy is 61.39%. From the interface, it is possible in one click to deploy the model and make API calls. We did not work further with this model.

##### B. Features Extraction + Pycaret / & Bag-Of-Words

At this point, we decided to take the data and follow what we had read in our preliminary research. We extracted various information about the texts such as sentence length, the number of stopwords per sentence, Part-Of-Speech statistics, and Entity Recognition to create a whole DataFrame of text metadata.

Only based on this metadata, we made a classification on the labels using the PyCaret library. PyCaret is an open-source, low-code machine learning library in Python. It allows, among other things, very easily and quickly to make comparisons between many different algorithms and do more advanced preprocessing on the data. The library also offers an easy way to save predictive models and deploy them.

On only the metadata, we obtained an accuracy of 53%. We were surprised in good, but we remain dissatisfied when we analyze the matrix confusion. Indeed, there is a too significant proportion of deviation and extreme on the wrong predictions. We have the intuition that the method is not good enough. We continue by trying to integrate Bag-Of-Words. The results are that... tbd.

##### C. CamemBERT For Sequence Classification

The CamemBERT model was proposed in CamemBERT: a Tasty French Language Model by Louis Martin, Benjamin Muller, Pedro Javier Ortiz Suárez, Yoann Dupont, Laurent Romary, Éric Villemonte de la Clergerie, Djamé Seddah, and Benoît Sagot. It is based on Facebook’s RoBERTa model released in 2019. (Huggingface, 2021). It is a model trained on 138GB of French text. We used more specifically "CamemBERTForSequenceClassification". CamemBERT Model transformer with a sequence classification/regression head on top. Our thought process is the following: with our knowledge in deep learning, why not take a model that already understands more or less French to make a classification. We could do this thanks to a tutorial from the author that has not left any more information on himself aside from his name is Olivier. (“Analyse de sentiments avec CamemBERT,” 2021)

Therefore, we had to transform the labels A1 to C2 into numbers from 0 to 5. Then, we had to do some text preprocessing via CamembertTokenizer to transform the data into tensors. As a result, we were able to train our PyTorch model via a GPU instance for about 20 epochs, about 5 hours. At each time, if the loss function improves, we register the model. This way, it is possible to make separate and multiple training sessions by reloading a model. When we make a prediction, we have to tokenize the text and then translate it from 0 to 5 again from A1 to C2.

The results of this model are phenomenal. Almost too good to be true. We reach 98% accuracy, and we observe a matrix confusion that is not far from the correct annotated difficulty when there is an error. Even stronger, when we check the incorrectly annotated sentences manually, we realize that perhaps the error comes from the quality of the annotation rather than the model.

CamemBERT - Classification Report

```python
                          precision    recall  f1-score   support

                       0       0.99      0.95      0.97       148
                       1       0.94      0.99      0.97       170
                       2       0.97      0.96      0.97       154
                       3       0.99      0.99      0.99       156
                       4       0.98      0.98      0.98       160
                       5       0.98      0.98      0.98       132

                accuracy                           0.98       920
               macro avg       0.98      0.98      0.98       920
            weighted avg       0.98      0.98      0.98       920
```

CamemBERT - Confusion Matrix

```python
                               [141,   6,   1,   0,   0,   0],
                               [  1, 168,   1,   0,   0,   0],
                               [  0,   4, 148,   2,   0,   0],
                               [  0,   0,   1, 155,   0,   0],
                               [  0,   0,   1,   0, 157,   2],
                               [  0,   0,   0,   0,   3, 129]
```

So we chose to take this model and deploy it.

### Deployment

For deployment, we created a simple Flask API that loads the model and predicts a sentence. The API is in the folder "api" of this GitHub repository. We then Dockerized the API and published it on Docker Hub. Afterward, we created an Azure Container instance, imported the Docker Flask API on Docker Hub and voilà.

A big lesson learned was that we could not run our docker container on clouds for a long time because we had Docker on the new macOS with Apple M1 chips. The Docker container architecture was in arm64, and it was not supported on Azure, and Google Cloud Run instances. It sounds simple, but it took a long time to understand because no error message logs understood and targeted this problem.

The team did not stop there, and we created a first frontend release in the folder of the same name in the GitHub repository or created a web application with the Python library "Streamlit" that we then hosted on Heroku. From this interface, it is possible to fill a sentence in a text input, and a request is sent to the API to have an answer in a very user-friendly and interactive way.

## 4. Bibliography

### Papers

The two main inspirations for the project :

- Curto, P., Mamede, N., & Baptista, J. (2015). Automatic Text Difficulty Classifier—Assisting the Selection Of Adequate Reading Materials For European Portuguese Teaching. 36‑44. https://doi.org/10.5220/0005428300360044
- Santucci, V., Santarelli, F., Forti, L., & Spina, S. (2020). Automatic Classification of Text Complexity. Applied Sciences, 10(20), 7285. https://doi.org/10.3390/app10207285

Other useful papers to correlate findings and provide additional insight :

- Balyan, R., McCarthy, K. S., & McNamara, D. S. (2018). Comparing Machine Learning Classification Approaches for Predicting Expository Text Difficulty. In Grantee Submission. https://eric.ed.gov/?id=ED585216
- Balyan, R., McCarthy, K. S., & McNamara, D. S. (2020). Applying Natural Language Processing and Hierarchical Machine Learning Approaches to Text Difficulty Classification. International Journal of Artificial Intelligence in Education, 30(3), 337‑370. https://doi.org/10.1007/s40593-020-00201-7
- Collins, E., Rozanov, N., & Zhang, B. (2018). Evolutionary Data Measures : Understanding the Difficulty of Text Classification Tasks. arXiv:1811.01910 [cs]. http://arxiv.org/abs/1811.01910
- Collins-Thompson, K., & Callan, J. (2004). A Language Modeling Approach to Predicting Reading Difficulty. 193‑200.
- Miltsakaki, E., & Troutt, A. (2008). Real Time Web Text Classification and Analysis of Reading Difficulty. Proceedings of the Third Workshop on Innovative Use of NLP for Building Educational Applications, 89‑97. https://www.aclweb.org/anthology/W08-0911
- Uchida, S., Takada, S., & Arase, Y. (2018, mai). CEFR-based Lexical Simplification Dataset. Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018). LREC 2018, Miyazaki, Japan. https://www.aclweb.org/anthology/L18-1514
- Wilkens, R., Zilio, L., & Fairon, C. (2018, mai). SW4ALL : A CEFR Classified and Aligned Corpus for Language Learning. Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018). LREC 2018, Miyazaki, Japan. https://www.aclweb.org/anthology/L18-1055

### Source of text data

- Niveaux—Delfdalf.ch. (2021). https://www.delfdalf.ch/niveaux

### Milestone 2
