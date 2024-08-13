# Comments-Indecency-Detector
Running head: COMMENTS INDECENCY DETECTOR 1

## COMMENTS INDECENCY DETECTOR

```
Yashika Singh, Yashaswi Rajesh Patki, Hang Yu, & Praveen Kumar Govind Reddy
Yeshiva University
```

```
Abstract
```
The proliferation of harmful and toxic speech on social media has escalated concerns about
online safety and the quality of digital discourse (Namdari, 2023). This study addresses the
urgent need for effective tools to identify and manage such content. We propose a multi-label
classification model designed to detect various forms of harmful speech in social media
comments, including toxic, severely toxic, obscene, threatening, insulting, and identity-
related harm (Zhang, Zhao, & LeCun, 2015). Utilizing natural language processing
techniques, our model leverages logistic regression extended through the
MultiOutputClassifier framework. The study demonstrates the model's effectiveness in
classifying diverse harmful behaviors and discusses its limitations and potential
improvements. This work contributes to enhancing online user safety and mitigating the
impact of harmful speech.


**Introduction**
The rise of social media has transformed communication, but it has also led to an
increase in harmful and toxic content (ElSherief, Kulkarni, & Wang, 2020). This trend has
sparked significant concern among researchers, policymakers, and platform administrators
about how to effectively manage and mitigate the spread of such content. The challenge lies
in developing robust systems that can accurately detect and classify various types of harmful
speech in real time. Addressing this issue is crucial for improving online interactions and
ensuring user safety.

This study aims to develop and evaluate a multi-label text classification model that
identifies different types of harmful comments on social media platforms. By focusing on
categories such as toxic, severely toxic, obscene, threatening, insulting, and identity-related
harm, the research combines advanced text preprocessing, feature extraction techniques, and
logistic regression to create a comprehensive solution. The use of the MultiOutputClassifier
framework and GridSearchCV for hyperparameter optimization enhances the model's
performance (Curry, Abercrombie, & Rieser, 2021). This research contributes to the broader
effort of refining online content moderation systems and fostering a safer digital
environment.

```
Background and Literature Review
```
**Previous Work on Content Moderation**

Content review has evolved from an early manual process to using complex digital
technology to solve problems (Kennedy, Bacon, Sahn, & von Vacano, 2020). Initially, content
review relied on rule-based systems that used keyword matching to filter out inappropriate
content. Although these methods are simple and effective, they are difficult to cope with the


complexity and subtle differences of natural language, resulting in frequent false positives
and omissions of harmful content.

With the advancement of machine learning, automatic content review systems have
emerged on the market, utilizing supervised learning techniques to improve accuracy. Early
technicians used models that utilized traditional classifiers such as Support Vector Machines
(SVM) and Naive Bayes to detect harmful content based on features extracted from text. The
introduction of deep learning methods, such as Convolutional Neural Networks (CNN) and
Recurrent Neural Networks (RNN), further enhances these systems by capturing finer
patterns and contextual information.

In recent years, multi label classification has become the key to solving the
complexity of modern content review. This method allows for the simultaneous detection of
various types of harmful behaviors, such as toxicity, obscenity, and threat
MultiOutputClassifier and other technologies achieve a more comprehensive evaluation by
predicting multiple labels for each content. Despite these advances, there are still ongoing
challenges in adapting to the constantly evolving language and balancing content regulation
with freedom of speech (ElSherief, Kulkarni, & Wang, 2020).

**Limitations of Existing Models**

Despite the advances in content moderation models, there are several limitations in
existing approaches that hinder their effectiveness and reliability. First, many traditional
models, including those based on simple keyword filtering or rule-based systems, lack the
ability to understand context and nuance in language. This often leads to high false positive
and false negative rates, where benign content may be flagged as harmful, and harmful
content may go undetected.


Second, the reliance on single-label classification models can be restrictive in a multi-
label environment like content moderation, where a single piece of content can exhibit
multiple types of harmful behavior simultaneously (Zhang, Zhao, & LeCun, 2015). Single-
label models typically treat each type of harmful behavior independently, ignoring potential
correlations between them. This can result in suboptimal performance, as the
interdependencies between different types of harmful content are not leveraged to improve
detection accuracy.

Furthermore, existing models often struggle with handling the vast and ever-evolving
variety of offensive language on the internet. Adversarial users continuously invent new
slang, euphemisms, and obfuscations to bypass moderation systems. Traditional models,
which may rely heavily on predefined vocabularies and fixed patterns, can quickly become
outdated and ineffective.

**Methodology**
To systematically describe our methodological process, Figure 1 illustrates the main
steps from data preparation to model evaluation. This flowchart outlines the key steps of the
entire method and demonstrates how each step is interconnected and influences each other.

Figure 1

_Methodology Flowchart for Multi-Label Classification Model_


_Note_. This figure outlines the method flow, including dataset preparation, preprocessing,
model training, and evaluation, with arrows showing the data processing sequence.

In the following sections, we will discuss in detail the specific content and
implementation methods of each step. Additionally, a brief overview of data analysis
techniques is provided to offer insight into any specific statistical methods applied beyond the
machine learning model itself.

**Dataset**

The dataset utilized in this study consists of a comprehensive collection of comments
from multiple sources, the following dataset was used by us as the training set:

YouTube Comments: 1000 hand-labeled comments, categorized into various toxicity
types including toxic, severe_toxic, obscene, threat, insult, identity_hate, abusive,


provocative, hatespeech, racist, nationalist, sexist, homophobic, religious hate, and radicalism
(Namdari, 2023).

Wikipedia Talk Pages: Comments from Wikipedia's talk pages, providing additional
examples of various toxicity types (Kaggle, 2018).

Facebook Dataset: 4,185 comments labeled for abuse binary, severity, directedness,
target group, and type (ElSherief, Kulkarni, & Wang, 2020). This dataset enhances the
model's ability to generalize across different social media platforms.

We use the following dataset as the test set:
Multi platform comment hybrid dataset: The dataset focuses on hate speech
measurement in English, encompassing a total of 135556 annotated social media comments
from Twitter, Reddit, and YouTube (Kennedy, Bacon, Sahn, & von Vacano, 2020).

**Data Preprocessing**

Data preprocessing is a crucial step in building an effective text classification model.
In this study, the preprocessing process includes the following key steps:

Text Cleaning: Remove HTML tags, URLs, user mentions (@username), hashtags
(#hashtag), and unnecessary punctuation. HTML tags are removed using the BeautifulSoup
library, while regular expressions are used to clear URLs and mentions.

Text Normalization: Convert text to lowercase and remove extra spaces. Regular
expressions are used to remove non-alphabetic characters, retaining the important text
information.

Emoji Handling: Convert emojis to descriptive text using the emoji library so that the
model can understand this information.


Tokenization and Lemmatization: Tokenize the text using the nltk library and apply a
lemmatizer (WordNetLemmatizer) to reduce words to their base forms. Common stop words
are removed to reduce noise.

Harmful Vocabulary Handling: Replace harmful vocabulary in the text based on
predefined regular expression patterns. These patterns include various forms of offensive,
insulting, and abusive words to ensure the model can recognize and handle such vocabulary.

Feature Extraction: Convert the processed text into TF-IDF features using
TfidfVectorizer for model training. The maximum number of features is set to 10,000 to
control the feature space size and improve training efficiency.

**Model Architecture**

To achieve multi-label text classification, the following model architecture is used:
Model Choice: A logistic regression model is used as the base classifier, combined
with MultiOutputClassifier for multi-label classification. L2 regularization is applied to the
logistic regression model to prevent overfitting.

Model Optimization: The logistic regression model is optimized using GridSearchCV.
The parameter grid includes different values for the regularization parameter C (0.01, 0.1, 1,
10, 100) to find the optimal regularization strength.

Model Training: Train the multi-label logistic regression model on the training set
using MultiOutputClassifier to handle each label as a separate binary classification problem
(Alsaedi & Shatnawi, 2020). Cross-validation is used during the training process to evaluate
model performance and optimize hyperparameters.

**Training and Evaluation**

```
The steps for training and evaluating the model include:
```

Data Splitting: Split the training dataset into training and validation sets to assess the
model's performance on unseen data. The split ratio is 80:20.

Training Process: Train the logistic regression model on the training data and use
GridSearchCV to select the best hyperparameter combination. The training process includes
feature extraction, model training, and hyperparameter optimization.

Performance Evaluation: Evaluate the model's performance on the validation set using
ROC AUC scores. The evaluation metrics include accuracy, precision, recall, and F1 score.
ROC AUC scores for each label are calculated to assess the model's multi-label classification
capability comprehensively.

Test Set Prediction: Generate predictions on the test set and save the results. The
predictions include the probability of each comment falling into each toxicity category.

```
Results
```
**Performance Metrics**

In this section, we evaluate the performance of the multi-label classification model
using various metrics: ROC AUC, accuracy, precision, recall, and F1 score. These metrics
were computed and analyzed for both the training and validation stages to assess the model's
effectiveness in detecting harmful comments.

ROC AUC (Receiver Operating Characteristic - Area Under Curve)
Validation ROC AUC: 0.
Explanation: The ROC AUC score of 0.9745 indicates that the model has a very high
ability to distinguish between harmful and non-harmful comments. An ROC AUC value close
to 1 suggests excellent performance in differentiating between positive and negative samples.


Accuracy
Validation Accuracy: 0.
Explanation: An accuracy of 0.9101 means that the model correctly classified
approximately 91.01% of the samples in the validation set. This measure reflects the overall
performance of the classifier, with a high accuracy indicating that most predictions are
correct.

Precision
Validation Precision: 0.
Explanation: A precision of 0.7084 means that approximately 70.84% of the samples
predicted as harmful by the model are indeed harmful. Precision measures the accuracy of the
positive predictions made by the model.

Recall
Validation Recall: 0.
Explanation: A recall of 0.4107 indicates that the model identified about 41.07% of
the actual harmful comments. Recall reflects the model's ability to detect all true positive
samples, and a lower recall may indicate difficulty in capturing all relevant cases.

F1 Score
Validation F1 Score: 0.
Explanation: An F1 score of 0.5026 is the harmonic mean of precision and recall,
providing a balance between the two metrics. A lower F1 score suggests that there is room for
improvement in both precision and recall.

**Output results presentation**


Table 1 presents the performance metrics for the multi-label classification model used
to detect harmful comments. The table includes key evaluation metrics such as ROC AUC,
accuracy, precision, recall, and F1 score. ROC AUC (Receiver Operating Characteristic -
Area Under Curve) measures the model's ability to distinguish between harmful and non-
harmful comments, with a validation score of 0.9745 indicating excellent performance.
Accuracy represents the overall proportion of correctly classified comments, achieving a high
value of 91.01%. Precision, at 0.7084, indicates that approximately 70.84% of the comments
predicted as harmful are indeed harmful. Recall, at 0.4107, shows that the model identified
about 41.07% of the actual harmful comments, highlighting an area for improvement in
detecting all relevant harmful content. The F1 score, which balances precision and recall, is
0.5026, reflecting the trade-off between the two metrics. This comprehensive evaluation
demonstrates the model's strengths and areas where further enhancement is needed.

Table 1

_Performance Metrics for Multi-Label Classification Model_

Performance Metric Value
Validation ROC AUC: 0.
Validation Accuracy: 0.
Validation Precision: 0.
Validation Recall: 0.
Validation F1 Score: 0.
_Note_. Table 1 presents the performance metrics for the multi-label classification model,
including ROC AUC, accuracy, precision, recall, and F1 score, all evaluated on the validation
dataset.


Table 2 displays the predicted probabilities for various toxicity categories, including
toxic, severely toxic, obscene, threat, insult, and identity hate, for a set of sample comments.
Each row represents the probability score assigned by the model for a specific category of
harmful behavior. By analyzing these probabilities, one can understand the model’s
predictions for different types of harmful content and assess its performance in classifying
complex and multi-faceted comments.

Table 2

_Predicted Probabilities for Different Toxicity Categories_

toxic Severe_toxic obscene threat insult Identity_hate comment_text
0.032944 0.005416 0.006737 0.003079 0.006721 0.008096 yes^ indeed sort reminds
elder lady played part...
0.141763 0.014296 0.018479 0.003066 0.020952 0.
trans woman reading tweet
right beautiful
0.380781 0.041821 0.094516 0.003267 0.084213 0.
question broad criticize
america country flee ...
0.068636 0.012653 0.014659 0.004228 0.029865 0.
time illegals go back
country origin keep free...
0.968883 0.049891 0.954708 0.023356 0.522066 0.019222 starter bend one pink kick
as pussy get taste ...
_Note_. Table 2 shows the predicted probabilities for each toxicity category for each comment.

**Summarize**

We have strengthened the main propositions and explanations put forward,
emphasizing the importance of this work beyond its direct scope. At the same time, the
existing research results provide valuable insights for the field of multi label text


classification, especially in detecting harmful online content. Future work can explore other
models and techniques to improve the recall and overall performance of classification
systems.

**Discussion**
This section discusses the strengths, challenges, limitations, and future directions of
the multi-label classification model for detecting harmful comments. The evaluation is based
on the metrics obtained from the model.

**Model Strengths**

High ROC AUC Score: The model achieved a validation ROC AUC score of 0.9745,
indicating a strong performance in distinguishing between harmful and non-harmful
comments. The ROC AUC metric suggests that the model effectively separates harmful
comments from non-harmful ones, providing a reliable indicator of the model's
discriminatory power.

High Accuracy: With a validation accuracy of 0.9101, the model correctly classified
91.01% of the comments. This high accuracy demonstrates that the model has a robust
overall performance, making it effective for real-world applications where precise
classification is crucial.

Comprehensive Evaluation with Multi-label Classification: The model’s ability to
handle multiple labels (toxic, severe_toxic, obscene, threat, insult, identity_hate) reflects its
robustness in detecting various types of harmful content. This multi-label capability ensures
that the model can address different aspects of harmful behavior, enhancing its applicability
in diverse scenarios.

**Challenges and Limitations**


Low Recall for Some Labels: The recall score of 0.4107 highlights that the model
struggles to identify all instances of harmful comments. Specifically, this suggests that the
model may miss a significant portion of true positive cases, leading to potential gaps in
detecting harmful content.

Imbalanced Precision and Recall: The precision (0.7084) and F1 score (0.5026)
indicate a trade-off between precision and recall. While the precision is relatively high, the
recall is comparatively low, which could result in a higher number of false negatives. This
imbalance shows that the model might not be capturing all relevant cases of harmful
comments effectively.

Performance on Long Texts: The model’s performance might vary based on the length
of the text. Longer comments or those with more complex structures might not be classified
as accurately as shorter, simpler ones. This limitation could affect the model’s overall
effectiveness in processing diverse comment types.

**Future Directions**

Improving Recall: Future work should focus on enhancing the recall of the model.
Techniques such as incorporating more diverse training data, adjusting class weights, or
exploring advanced model architectures could help in capturing more true positive cases and
improving recall.

Addressing Text Length Variation: Investigating the impact of text length on model
performance and applying text preprocessing techniques or using models better suited for
varying text lengths could improve classification accuracy for longer comments.

Expanding the Dataset: Increasing the size and diversity of the dataset could provide a
more comprehensive representation of harmful comments. This expansion will include data


from different sources or more examples of rare harmful categories to enhance model training
and generalization.

Explore advanced techniques: Implementing more sophisticated techniques, using
better pre-trained language models for transfer learning or experimenting with neural network
architectures can improve classification performance and better handle complex comment
structures.

Enhance feature engineering: Further explore feature engineering, perform semantic
analysis or context embedding in more detail, provide more meaningful comment
representations, and improve the model's ability to detect harmful content.

```
Conclusion
```
**Summary of Contributions**

This study focused on developing and evaluating a multi-label classification model
for detecting harmful comments on social media. The model used logistic regression and TF-
IDF feature extraction and GridSearchCV for parameter optimization. The main contributions
include:

Novel model development: A powerful classification model was developed to identify
various types of harmful content, including toxic, severely toxic, obscene, threatening,
insulting, and identity hate comments. The model was able to handle multiple labels,
reflecting its versatility in detecting different aspects of harmful behavior.

Better performance: The model achieved a validation ROC AUC score of 0.9745,
showing high discriminative ability in distinguishing between harmful and harmless
comments. The validation accuracy of 0.9101 further confirmed the effectiveness of the
model in correctly classifying comments.


Provide reference for preprocessing and optimization techniques for related project
data: This study combined comprehensive text preprocessing, including HTML tag removal,
URL processing, emoji conversion, and tokenization. Hyperparameter tuning using
GridSearchCV ensured that model parameters were optimized for better performance.

**Implications for Social Media Platforms**

The results of this study have several implications for social media platforms:
Improving content moderation: The model is able to classify harmful comments with
high precision and ROC AUC scores, which can help social media platforms automate
content moderation. By integrating such models into their systems through technology,
platforms can better detect and manage harmful content, thereby enhancing user experience
and security.

Addressing a variety of harmful content: The model is able to identify multiple types
of harmful behaviors, providing a comprehensive tool for addressing various forms of
negative interactions. This helps create a safer online environment by addressing different
aspects of harmful content.

Guidelines for future model improvements: The insights gained from the model's
performance, including its strengths and areas for improvement, can guide the future
development of harmful content detection. Platforms can use these insights to improve their
models, incorporate advanced techniques, and address challenges such as low recall and text
length variation.


```
References
```
Namdari, R. (2023). _YouTube toxicity data_ [Data set]. Kaggle.
https://www.kaggle.com/datasets/reihanenamdari/youtube-toxicity-data

Zhang, L., Zhao, J., & LeCun, Y. (2015). Character-level convolutional networks for text
classification. _Proceedings of the 28th International Conference on Neural
Information Processing Systems (NeurIPS 2015)_ , 649-657.
https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-
classification.pdf

Badjatiya, P., Gupta, S., Gupta, M., & Varma, V. (2017). Deep learning for hate speech
detection in tweets. _Proceedings of the 26th International Conference on World Wide
Web (WWW 2017)_ , 759-760.
https://dl.acm.org/doi/10.1145/3038912.

Garg, N., & Lakkaraju, H. (2019). A dataset and a baseline for hate speech detection in social
media. _Proceedings of the 2019 Conference of the North American Chapter of the
Association for Computational Linguistics (NAACL 2019)_ , 46-53.
https://www.aclweb.org/anthology/N19- 1006

ElSherief, M., Kulkarni, V., & Wang, C. (2020). Hate speech detection and the problem of
offensive language_. Proceedings of the 2020 Conference on Empirical Methods in
Natural Language Processing (EMNLP 2020)_ , 145-158.
https://www.aclweb.org/anthology/2020.emnlp-main.

Alsaedi, A., & Shatnawi, S. (2020). Multi-label text classification using deep learning: A
comprehensive survey. _Proceedings of the 2020 IEEE International Conference on
Big Data (Big Data)_ , 1247-1256.
https://doi.org/10.1109/BigData50022.2020.


Kaggle. (2018). _Jigsaw toxic comment classification challenge_ [Data set]. Kaggle.
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data

Curry, A. C., Abercrombie, G., & Rieser, V. (2021). ConvAbuse: Data, analysis, and
benchmarks for nuanced detection in conversational AI. In _Proceedings of the 2021
Conference on Empirical Methods in Natural Language Processing_ (pp. 7388-7403).
https://aclanthology.org/2021.emnlp-main.587/

Kennedy, C. J., Bacon, G., Sahn, A., & von Vacano, C. (2020). Constructing interval
variables via faceted Rasch measurement and multitask deep learning: A hate speech
application. _arXiv preprint arXiv:2009._.
https://arxiv.org/abs/2009.



