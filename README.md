# Comments-Indecency-Detector
This repository contains a Python implementation of a custom tokenizer designed to preprocess and clean text data, particularly focusing on handling offensive language, repetitive characters, and emojis. The code is built to be used in NLP tasks such as sentiment analysis, where preprocessing is crucial for improving model performance.

Key Features:
Pattern Replacement: Utilizes a dictionary of regular expressions to identify and replace offensive words and phrases with more neutral terms.
Text Cleaning: Removes HTML tags, URLs, mentions, hashtags, and non-alphabetic characters from the text.
Emoji Handling: Converts emojis into descriptive text using the emoji library, allowing for more consistent analysis of text data that includes emojis.
Repetition Removal: Reduces sequences of repeated characters to a single character to standardize the text.
Tokenization and Lemmatization: Tokenizes the text into words and applies lemmatization to reduce words to their base forms. It also filters out common English stopwords.
Customizable: The tokenizer class (PatternTokenizer) can be easily modified to adjust the patterns, filters, and preprocessing steps according to specific needs.
Usage:
Input: The code takes raw text data from CSV files, processes the comment_text column, and outputs the cleaned and tokenized text.
Output: The processed text is saved as CSV files, ready for further NLP tasks such as training machine learning models.
Example:
This repository includes a main() function that demonstrates how to use the PatternTokenizer to preprocess and save the train and test datasets.

Dependencies:
1. pandas
2. nltk
3. beautifulsoup4
4. emoji
