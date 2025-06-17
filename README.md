# loan-risk-modeling
Start of Preprocessing_and_analysis
Data was gathered from this Kaggle page https://www.kaggle.com/datasets/wordsforthewise/lending-club
In this project I will model the risk of a loan after loan acceptance. 
I created a binary classification called is_defult that includes "defulated", "charged_off", "does not meet credit policy: charged_off" 
Numerical variables were studies: interest rate and fico scores were stronger indicators. Fico is faced with survivorship bias since very low fico scores were dropped which means they are not included in the scope of this project. Interest rate needs to be used carfully since it can arbitrarily raise the accuracy of the model due to the high correlation with the target variable. 
The other side of the binary target are "paid off" loans and "Does not meet the credit policy. Status:Fully Paid" 
The rest of the types of loans not mentioned but that also appear in the bar chart below.
![image](https://github.com/user-attachments/assets/a14166c1-648b-4545-b06c-840de27b7608)
Before that I also looked at the correlation between the numerical variables shown bellow 
![image](https://github.com/user-attachments/assets/1d251685-ec3f-4826-be5e-23eed427d50c)
interest rates have a negative correlation with fico scores meaning that the higher your score, the lower your obligated interest payment. 
"revo_rate" or the amount of credit used was negativlely correlated with fico. The less credit you use the better your fico score. 
Finnaly, instalments have a near 1:1 with loan amount (which is 1:1 with funded amount), which means that the higher your loan the larger your installment payments.
End of Preprocessing_and_analysis

### Logistic Regression – Baseline (Unbalanced)
- ROC AUC: 0.69
- Recall (default): 6%
- Comment: Poor default prediction

### Logistic Regression - SMOTE
- ROC AUC: 0.69
- Recall (default): 61%
- Comment: Better target class identification

The model is catching 6–10× more defaulters than before
The model is no longer ignoring risk to protect accuracy
The F1 for default(target) went from 0.10 → 0.41
I used SMOTE (Synthetic Minority Over-sampling Technique)
Becuase it actually creates synthetic new samples in the minority class
These are interpolations between real data points, not random
Model gets to learn more diverse decision boundaries for defaults
So SMOTE doesn't just care more about defaults — it helps the model see more of them.

