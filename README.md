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

Beginning on Autoencoders
I attempted to use categorical variables in the model by turning them into true/false columns using autoencoders. The result was a model that took very long to run, mostly failed to converge and resulted in lower performance 

### Logistic Regression – Autoencoded Categorical Features
- ROC AUC: 0.68
- Recall (default): 35%
- Comment: Features lowered performance and increased training time.

  End of autoencoders

![image](https://github.com/user-attachments/assets/f913014a-28a8-484f-811a-4b539f5cd0d9)
![image](https://github.com/user-attachments/assets/d162fa2e-0ded-41f5-8391-3806d437e945)

Interest payment has the highest effect because as risk increases towards default interest rate is increased to cover the risk. 
This is why it is the primary variable for the model is interest because it refelect back what was established in the risk analysis which lead to the stats of the approved loan. 

Currently the model could work as an extra layer of analyis applied to loans to decrease the ammount of loans distributed but the ammount of money saved by banks since the model can dedect defaulters to 60% accuracy. I saw that using a nueral-network will increase that accuracy towards 90% and will be used.

But the goal is to increase the number of loans distributed, making loans more available to people in need while making it profitable for lenders. I have some steps to do this: 

1. Convert the model to be trained using the variables found in the unapproved loans dataset. Train it on the approved dataset. 

2. Assume that SMOTE will cover the difference in variable distribution. Use the model on the unapproved dataset. 

3. To validate the study, we need to find some way of turning the validation dataset, ie. unapproved loans, into a supervised problem by accurately simulating defaulters and non-defaulters and see if the model can perform at picking out the few non-defaulters. 

4. When accuracy and recall are satisfiable, we will collect more data and add it to the training and validation datasets to make the model more generalized to the real world. Additionally, we could make a model for types of lenders such as banks, lending clubs, or car retailers. 

5. The model will be turned into an API for the final step where it will take a single unapproved user and their inputs and will predict the probability that they could be a non-defaulter. 

There are only three variables that match between both datasets. 

![image](https://github.com/user-attachments/assets/ac116319-b441-4635-953c-b14cd0ef9b3b)

![image](https://github.com/user-attachments/assets/3db321c7-f530-405f-a574-f9f9c093e034)

![image](https://github.com/user-attachments/assets/b15d3289-0883-4965-804c-db4bcb31553f)

![image](https://github.com/user-attachments/assets/add3a75f-1b35-4d68-b5f2-5cfafd0ba288)


  
