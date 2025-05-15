# Language Detection System Report

## Model Performance
- Accuracy: 98.60%

## Classification Report
```
              precision    recall  f1-score   support

     English       0.97      0.97      0.97        39
      French       0.98      1.00      0.99        40
      German       0.98      0.98      0.98        48
    Japanese       1.00      1.00      1.00        36
      Korean       1.00      1.00      1.00         7
     Spanish       1.00      0.80      0.89         5
  Vietnamese       1.00      1.00      1.00        39

    accuracy                           0.99       214
   macro avg       0.99      0.96      0.98       214
weighted avg       0.99      0.99      0.99       214


## Top Features by Language

### English
- l: -4.68
- is: -4.64
- w: -4.63
- y: -4.42
- e : -4.38
- s : -4.37
- n: -4.36
-  i: -4.34
- r: -4.32
-  t: -4.26
- th: -4.23
- he: -4.20
- h: -4.16
- a: -4.13
- s: -4.10
- i: -4.04
- t: -3.99
- o: -3.87
- e: -3.60
-  : -2.63

### French
- v: -4.68
- ou: -4.68
-  e: -4.66
- le: -4.58
- é: -4.57
- n: -4.52
- p: -4.50
- es: -4.41
-  l: -4.40
- i: -4.36
- t: -4.24
- u: -4.19
- o: -4.18
- e : -4.14
- r: -4.10
- a: -4.07
- s: -4.00
- l: -3.99
- e: -3.36
-  : -2.79

### German
- .: -4.57
- l: -4.55
- ie: -4.54
- u: -4.52
- t : -4.52
-  i: -4.42
- c: -4.39
- a: -4.35
- er: -4.30
-  d: -4.27
- d: -4.24
- h: -4.24
- ch: -4.19
- n: -4.15
- r: -4.13
- t: -4.01
- s: -3.90
- i: -3.80
- e: -3.43
-  : -2.73

### Japanese
- ？ : -4.65
- か？: -4.65
- ？: -4.65
- いま: -4.49
- し: -4.44
- てい: -4.42
- か: -4.34
- は: -4.32
- て: -4.30
- です: -4.05
- ます: -4.03
- で: -3.95
- い: -3.87
- ま: -3.85
-  : -3.83
- が: -3.82
- す。: -3.71
- 。 : -3.65
- 。: -3.63
- す: -3.50

### Korean
- 를 : -4.97
- 를: -4.97
- 로: -4.92
- 하: -4.89
- 이 : -4.87
- 습: -4.86
- 있: -4.85
- 고: -4.80
- 을: -4.75
- 을 : -4.75
- 에: -4.74
- 나: -4.70
- 니다: -4.60
- 니: -4.59
- 는 : -4.55
- 는: -4.48
- 이: -4.32
- 다.: -4.11
- 다: -4.04
-  : -2.92

### Spanish
- d: -4.82
- en: -4.75
- es: -4.66
- i: -4.64
- t: -4.63
- u: -4.61
- c: -4.60
- p: -4.56
- o : -4.55
- l: -4.55
-  e: -4.50
- m: -4.47
- s: -4.33
- n: -4.25
- a : -4.23
- r: -4.12
- o: -3.89
- a: -3.68
- e: -3.67
-  : -2.97

### Vietnamese
- . : -4.82
- .: -4.82
- à: -4.76
- n : -4.67
-  n: -4.66
- g : -4.63
- i: -4.62
-  c: -4.59
- i : -4.51
- c: -4.51
- ô: -4.50
- t: -4.38
- ng: -4.32
- g: -4.27
-  t: -4.26
- h: -4.23
-  đ: -4.18
- đ: -4.18
- n: -3.91
-  : -2.65

## Theoretical Background

### Multinomial Naive Bayes
The Multinomial Naive Bayes classifier is particularly suitable for text classification tasks. It works by:
1. Calculating the probability of each word occurring in each language
2. Using these probabilities to determine the most likely language for a given text
3. Taking into account the frequency of words (bag-of-words approach)

### Role of Word Frequency
- Word frequency plays a crucial role in language detection
- Common words and their patterns are unique to each language
- The model learns these patterns during training
- More training data leads to better recognition of language-specific patterns
