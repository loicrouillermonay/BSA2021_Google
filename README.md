# Big-Scale Analytics 2021 - Team Google

- David Netter
- Francis Ruckstuhl
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


### Milestone 3 - Iterate/improve & AIcrowd

The purpose of this milestone is to improve the models and make submissions to the AIcrowd private competition. In parallel to that, the UI and the documentation are improved.


## 2. Approach to solve the problem

To solve the problem of creating a predictive model, we will use a method inspired from the papers we have read and learned in the research findings mentioned at the end of the readme. The method is described in the following paragraphs.

First of all, we collected data from reading comprehension tests from samples of exams made available by the "Fondation Esprit Francophonie DELF DALF Suisse". Those texts are already labeled according to their difficulty level. This allowed us to collect a whole set of more than 1000 sentences sorted by experts from A1 level to C2. Therefore, the quality and the accuracy of the training data is guaranteed.

Following this, we will create an NLP Pipeline to transform the sentences into the most efficient features possible to train a machine learning model. To boost accuracy, we will have to extract additional features such as measuring the number of frequencies of words from 1-4 syllables per sentence, calculate the diversity of vocabulary per sentence with specific formulas or external python libraries such as "wordstats", count the length of each sentence and use various lexical treatments common to NLP such as lemmatization. More on this subject can be found highlighted in yellow in the "research findings" Microsoft Word file in the "documents" folder of this repository.

Thus, it has been decided that we will treat this assignment as a classification problem and we will take into consideration each sentence as a whole for the difficulty classification. However, we have the intuition that it would be possible to treat it as a regression problem; with a difficulty that could range from 1 to 6, such as from A1 to C2, it would be possible to create a model that can then evaluate a slight difference in level between two texts of the same level as long as a result is higher. For example, two texts with a model that predicts difficulties of 5.16 and 5.38 will be texts judged as C1, but the second would be slightly higher and could offer a more precise system of recommendations to people. However, we will not choose this method. Or, at least, not at this time in the project.

Regarding the creation of the model, we will use the resources made available on the Google Cloud. We will first upload the data to the Google Cloud Platform and try to work with Google AutoML; then, if we feel that we are not satisfied with the solution, it would be possible to do things a more manually by creating an instance of AI Platform Notebooks (JupyterLab). It is a virtual machine instance that comes preinstalled with the latest machine learning and data science libraries. If that is the case, we will evaluate algorithms that performed well in the papers we read, such as those highlighted in blue in the research findings Microsoft Word file in the "documents" folder of this repository. Some of those are Logistic Regression, AdaBoost, LDA, SVM, kNN, and Neural Networks.

Then we will have to optimize the model, save the best performing model and deploy it. Those stages will be described in greater detail later in the course of the project.


## 3. Contribution

Team Google annotated 1020 sentences for Milestone 1.


## 4. Synthesis of the work done on Milestone 2

A lot of work was performed for milestone two, so this chapter will summarize it as concisely as possible. Firstly, it will describe our strategy regarding the training data. Secondly, it will then explain the simultaneous creation of three types of models: Google Cloud Platform, Features extraction + Pycaret & BOW and CamemBERT. Lastly, it will cover how the model was deployed as an API with a user friendly UI frontend.


### 4.1 A quick word on the data

To make it simpler for us, what is described in the following chapter has been first tested and implemented on our own dataset. However, the amount of data proved too insufficient to conduct significant operations. In the end, this method mainly allowed us to find which model underfitted the least.

That is why we switched strategy afterwards and used 9174 observations. They are the result of a merge of our colleagues' data that we found was the most qualitative mixed with ours. This allowed us to "get down to business" and work on models that were not constantly underfitting. This way, we could make a better educated guess about which model was to be chosen for the final milestone of this project, where everyone will have the same dataset to train.


### 4.2 Predictive models

Three types of models were created in parallel to evaluate the results and choose the best one. Here are the basic metrics for them:

- Base rate: 0.16
- Default Rate: 0.19

_NOTA BENE: All the notebooks for each of the models as well as the one for data preparation can be found in the "notebooks" folder in the GitHub archive._


#### A. Google Cloud Platform - Natural Language

This is the simplest solution. All data composed of a text column and a column with the labels were uploaded to Google Storage. It was then transmitted to the "Natural Language" application. There, Google creates a model on its own via its text classification wizard. The training lasts one day and the model's accuracy is 61.39%.

From the Google Cloud Platform, there is the possibility to deploy the model and make API calls in one click. There was no fine-tuning possible and this model was kept as a backup if no other predictive models could be created. It also served as a kind of "benchmark" to judge of the findings with the other models.


#### B. Features Extraction + Pycaret / & Bag-Of-Words

Here, we applied the methods explained in our preliminary search. Various features were extracted about the texts such as sentence length, the number of stopwords per sentence, Part-Of-Speech statistics, and Entity Recognition. They were all put in a DataFrame of text metadata.

Based on that, we used the PyCaret library to classify the labels. It is an open-source, low-code machine learning library in Python. It allows, among many other things, to easily compare different alorithms and perform more advanced preprocessing on the data. The library also offers an easy way to save predictive models and later deploy them.

As a result of using the metadata, the best result was obtained from the random forest classifier with an accuracy of 53%. We were positively surprised, altough dissatisfied by the confusion matrix. Indeed, the proportion of deviation and extremes on the wrong predictions is too significant. We have the intuition that the method is not good enough and shouldn't be further refined.

```Python
Model                           Accuracy   AUC      Recall    Prec.     TT(Sec)
Random Forest Classifier        0.5302     0.8410   0.5314    0.5295    3.108
Extra Trees Classifier          0.5300     0.8441   0.5315    0.5304    3.510
CatBoost Classifier             0.5029     0.8136   0.5050    0.5011    35.704
K Neighbors Classifier          0.3935     0.7193   0.3988    0.3894    1.712
Logistic Regression             0.3667     0.7470   0.3760    0.3613    2.044
```

The use of the Bag-Of-Words technique was tried for a week but proved impractical due to the explosion of the observation space and the difficulty of training a model, even using PCA. We decided to use only the features extracted from the text.


#### C. CamemBERT For Sequence Classification

The CamemBERT model was proposed in the paper "CamemBERT: a Tasty French Language Model" by Louis Martin, Benjamin Muller, Pedro Javier Ortiz Suárez, Yoann Dupont, Laurent Romary, Éric Villemonte de la Clergerie, Djamé Seddah, and Benoît Sagot (2020). It was created based on Facebook’s RoBERTa model, released in 2019. (Huggingface, 2021). It is a model trained on 138GB of French text. For our project, we more specifically used "CamemBERTForSequenceClassification": it is the CamemBERT Model transformer with a sequence classification/regression head on top. Our thought process is the following: with our knowledge in deep learning, why not take a model that is already pretrained with French to make a classification ? We could do it thanks to a very clear tutorial from a person named Oliver. He has not left any more information on himself aside from his name and his tutorial: “Analyse de sentiments avec CamemBERT,” 2021.

To use CamemBERT, labels were transformed into numbers. A1 to C2 became 0 to 5. Then, we performed text preprocessing via CamembertTokenizer to transform the data into tensors. As a result, we were able to train the model via a GPU instance for about 20 epochs, about 5 hours. At each interval, if the loss function improves, the model is saved. This way, it is possible to make separate and multiple training sessions by reloading a model. When a prediction is made, the text has to be tokenized and then transposed again, this time from "0 to 5" to "A1 to C2".

The results of this model are excellent: _98% accuracy_ !! We insist on the fact that we double checked that the predictions were made on the 10% of the dataset that the model had never seen. In the confusion matrix, we also observe that when there is an error, it is because the model predicted that the sentence was from an immediately adjacent level. Furthermore, when we check the incorrectly annotated sentences manually, we realize that perhaps the error comes from the quality of the annotation rather than the model.

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

This model is the one we deployed and the one we will do the final training with.


### 4.3 Deployment

For the deployment of the model, a simple Flask API was created. In effect, it loads the model and predicts a sentence when receiving a request. The API was Dockerized and published on Docker Hub. We imported this Docker Container regrouping the Flask API with the model on an Azure Container created for this purpose. The API is located at the public address http://51.103.156.182/. Predictions can be made through a request with the "text" query param as KEY and the sentence as VALUE on the address: http://51.103.156.182/api/predict. Be careful, the container does not run constantly to avoid superfluous costs and the public address may be updated/changed in the future. If this is the case and there is a need to test it, you should write us an email, we will quickly respond to any of your enquiries.

_NOTA BENE: The API and documentation (ReadMe) on how to use it can be found in the "api" folder of this GitHub repository._

The team did not stop there: we also released a frontend with UI called "Lingorank UI". It can be found in the folder of the same name in the GitHub repository. It was created with the Python library "Streamlit", making it easy to code an MVP, and the application is hosted on Heroku. From this interface, it is possible to write either sole sentences or whole texts in a "text input area". Through that, a request is sent to the API to have an answer in a very user-friendly and interactive way.

https://lingorank-frontend.herokuapp.com/

A big lesson learned during this phase was that we could not make our docker container run on any clouds. This was due to the usage of the new macOS with Apple M1 chips. The Docker container architecture was in arm64, and it was not supported on Azure, and Google Cloud Run instances. It sounds simple, but it took a long time to understand because the error and message logs did not targeted this problem effectively. A different PC was then used to create a docker container and get around this problem.


## 5. Synthesis of the work done on Milestone 3

A folder named aicrowd has been created where all the models have been adapted for the competition. There, the results of the submissions are gathered, as well as the notebooks used to prepare the data and train the models.


### 5.1 CamemBERT on AIcrowd

There is not enough data to effectively train this model in the aicrowd competition. However, after multiple attempts, we noticed that the results were better when we had a smaller batch size, and no text processing at all.


### 5.2 Cognates + Features Extraction + Pycaret

All sentences have been translated into English using the Google Translate API. Then, multiple algorithms of distance calculation between the difference of strings were applied to the whole sentences. These elements were then added to the data used in the Pycaret model, and we notice an increase in accuracy. The cognates are, therefore, helpful. However, based on the results of aicrowd, we are still far from the performance of CamemBERT.

```Python
Model                           Accuracy   AUC      Recall    Prec.     TT(Sec)
Extra Trees Classifier          0.5629     0.8613   0.5694    0.5605    1.392
Random Forest Classifier        0.5577     0.8563   0.5624    0.5504    1.690
CatBoost Classifier             0.5349     0.8292   0.5367    0.5311    27.454
Light Gradient Boosting Machine 0.5276     0.8279   0.5290    0.5294    1.596
Decision Tree Classifier        0.4781     0.6879   0.4779    0.4774    0.138
```


### 5.3 Lingorank Web App

The UI has been improved and now displays the perceived difficulty of each word by the model. The difficulty levels of words are usually low and far from the predicted difficulty of the sentence because this process does not take into account the relationship between the words.


### 5.4 Next steps

The next steps are to train a few different versions of the models to send to AIcrowd to see potential improvements in our score, improve the UI by adding a tab to visualize the intensity of cognates within a sentence, adjust the project according to the evaluation criteria and prepare the final presentation.


## 6. Last Milestone : Conclusion of the project & General administrative information

### 6.1 General Architecture

You can find in the figure below the general architecture of this project. 
- For the Backend, data is first processed by the CamemBERT model. It can then be interacted with thanks to a Flask API that was dockerized. 
- For the frontend, the web app is developed with the Streamlit library and hosted on heroku. 
- Backend and Frontend are then hosted on an Azure cloud container to be available online.

# **Insert Figure**


### 6.2 Instructions from A to Z on how to train, deploy and use the model

_Nota Bene: This chapter aims to summarize in a few paragraphs what has been explained until now. It's normal if there is recurring information.


#### Backend

1) It all starts with a dataset. In it, two columns: one with sentences in French and the other with a label that goes from A1 to C2 to represent the difficulty of each of those sentences. All of that is in csv format. This dataset will have to be used with the notebook found in ```/notebook/BSA2021_Google_LingoRank_with_CamemBERT.ipynb```. 
In this notebook, you only need to connect to your google drive, import the data in csv format, specify the amount of _epochs_ the model should train and then wait for the end of the training. After that, it will be automaticaly stored in your Google drive if its performance improved during the training compared to the previous ones. The notebook will automatically tokenize the sentences in a format accepted by the CamemBERT architecture that we use. The estimated time to train the model is of around 5-6 hours.

2) Afterwards, some very precise commands in the terminal will have to be performed to setup for future operations. All these commands are documented very clearly in the file at the root of the project named ```DOCUMENTATION.md```.

3) The next step is to take this trained model (here named ```lingorank.pt```) and create an API for it. This is a normal Flask API that loads the model and then uses it to predict the difficulty on two endpoints: either for a complete sentence (as originally intended), or either for each of its words (as required for this project).

_Nota bene: the model weights more than 400Mo, and therefore can't be uploaded on GitHub. The trained model we use can be downloaded at this link: https://drive.switch.ch/index.php/s/6njoQoM0R3ML5kf and must be placed in the ```/API``` folder._

4) After that, we have to dockerize the API to host it on Azure and have a deployed, accessible endpoint. So, we created a dockerfile that has two roles: to install the necessary dependencies for the API, to expose the port 80 (which is the default port for Azure to make requests). Once this is done, we just have to build the container with the commands that can be found in the documentation (the ```DOCUMENTATION.md``` file). After that, to make it easier to handle, we upload the container on Docker Hub in order to retrieve it easily on Azure.

5) On Azure, thanks to the student offer, we have the necessary credits to create a container instance. To do so, we take the link of our container on Docker Hub and we reserve a 4 GB container instance for it on Azure. They then take care of downloading our Docker container and starting it. Afterwards, they give us an IP address to which we can make API calls.


#### Frontend

1) For the frontend, we use the open-source library 'streamlit', which allows to easily create data science web applications. A simple, yet efficient layout can be built in a few lines to allow users to enter French sentences. Those sentences are then sent to the API, which will return the result their difficulty and show it on the screen for the users. Additionnally, right after that, another API call is sent to retrieve the difficulty of each word, which will be display according to a color code on the web page.

2) Once the web interface is developed, we decided to use Heroku (a PaaS) to host the frontend for free thanks to their Free-tier service. In a few commands in the terminal which are also listed in ```DOCUMENTATION.md```, the ```/frontend``` folder as well as all the necessary dependencies are installed and then deployed in a Heroku style container (in the same way as Docker) and are put online at the address: https://lingorank-frontend.herokuapp.com/


### 6.3 Model and Model Evaluation overal summary

#### Predictive model description

We use the power of Deep Learning and especially Transfer Learning. This means that we take a model that has been trained on a huge data set and fine-tune it to a specific task. So we chose CamemBERT, an iteration of the BERT (Bidirectional Encoder Representations from Transformers) architecture, which is a model made by researchers at Google. CamemBERT was presented in the paper _CamemBERT: a Tasty French Language Model by Louis Martin, Benjamin Muller, Pedro Javier Ortiz Suárez, Yoann Dupont, Laurent Romary, Éric Villemonte de la Clergerie, Djamé Seddah, and Benoît Sagot_. This model was trained on 138GB of French text.
With the help of a blog post by Olivier (2021) and the Huggingface Transformers library, we have adapted this model and added a classification layer with 6 outputs at the end of it. Thus, it becomes possible to train it to classify French sentences difficulty levels from A1 to C2.


#### A word about combining multiple models

In our approach with Pycaret, we combined several models. We used two methods: ensembling (boosting and bragging) and stacking (use several different algorithms and we average the each of their output). It was possible to do boosting and bragging very easily with this open source PYcaret library: https://www.upgrad.com/blog/bagging-vs-boosting/#:~:text=Bagging%20and%20Boosting%3A%20Differences&text=Bagging%20is%20a%20method%20of,Boosting%20decreases%20bias%2C%20not%20variance. 
However, the results were not fundamentally superior. The predictions took longer to perform without being really more accurate.
For the stacking, we used several models. That is, we took the top 3 machine learning algorithms with the best results on our dataset and then averaged their three predictions to assign the label. This technique was not that notably successful either successful.
So, in essence, combining models lead to improvements, but they were far from major.



## 7. Bibliography

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

- Google Cloud. (2021). Troubleshooting  |  Cloud Run Documentation  |  Google Cloud. https://cloud.google.com/run/docs/troubleshooting
- Huggingface. (2021). CamemBERT. model_doc/camembert.html
- Martin, L., Muller, B., Suárez, P. J. O., Dupont, Y., Romary, L., de la Clergerie, É. V., Seddah, D., & Sagot, B. (2020). CamemBERT : A Tasty French Language Model. Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 7203‑7219. https://doi.org/10.18653/v1/2020.acl-main.645
- Mahugh, D. (2020, February 17). Deploying a Flask app to Google App Engine. Medium. https://medium.com/@dmahugh_70618/deploying-a-flask-app-to-google-app-engine-faa883b5ffab
- Nakamura, T. N. (2020, November 5). How to deploy a simple Flask app on Cloud Run with Cloud Endpoint. Medium. https://medium.com/fullstackai/how-to-deploy-a-simple-flask-app-on-cloud-run-with-cloud-endpoint-e10088170eb7
- Olivier. (2021, January 5). Analyse de sentiments avec CamemBERT. Le Data Scientist. https://ledatascientist.com/analyse-de-sentiments-avec-camembert/
- Reynoso, R. (2020, July 15). How to Deploy Multiple Apps Under a Single GitHub Repository to Heroku. Medium. https://betterprogramming.pub/how-to-deploy-multiple-apps-under-a-single-github-repository-to-heroku-f6177489d38
- Sanchez, A. (2018, April 8). Creating and Deploying a Flask app with Docker on Azure in 5 Easy Steps. Medium. https://medium.com/@alexjsanchez/creating-and-deploying-a-flask-app-with-docker-on-azure-in-5-easy-9f7aa7a12145
- Shchutski, V. (2020, February 11). French NLP: Entamez le CamemBERT avec les librairies fast-bert et transformers. Medium. https://medium.com/@vitalshchutski/french-nlp-entamez-le-camembert-avec-les-librairies-fast-bert-et-transformers-14e65f84c148
- Vogel, J. (2021, January 30). How to Actually Deploy Docker Images Built on M1 Macs With Apple Silicon. Medium. https://betterprogramming.pub/how-to-actually-deploy-docker-images-built-on-a-m1-macs-with-apple-silicon-a35e39318e97

```

```
