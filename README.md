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
I also decided I might need to only take people who were close to the risk/non-approval line to train a model more specific to the type of people who got rejected. So, knowing the distribution of loan grade is useful
![image](https://github.com/user-attachments/assets/f98303af-7e77-4fdd-bcfd-84efc75059f7)

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

Using a model to assing probabilities of default to unapproved members resulted in a highly biased result:
![image](https://github.com/user-attachments/assets/1410b97c-c741-4fc2-aae5-ff05d7c3e153)
Becuase the model is trained to assing mostly non-defaults which is what I thought might happen. Even with a resampling technique (used Standard_Scalar) the problem is that the model will still asign mostly non-defaults. That means when applied to a model, a random forrest model was used 
![image](https://github.com/user-attachments/assets/7e6cf0de-ce82-4f11-b019-c10ed01c7f27)
the results are completly biased. 

Now moving forward 
1: I'll refocus the model on low level loans like D G and F which will be more similar to the unapproved cases then move to clustering and find unapproved users who were similar to approved and non-defaulted users which can make a case that they could have been good investments. 2: Then we can a plot new users onto the graph and if they are in the "accepted zone" they can be classifed as passers. On the graph there will be three colors: 1 for unnaproved people, 2: approved and non-defaulted people, 3: approved and defaulted people. 3: Then I want to build a better version of the risk model but don't have much of a strategy for that yet. 
Here’s a high-level roadmap for the API idea:

Cluster rejected applicants using unsupervised learning (e.g. DBSCAN, Gaussian Mixture Models, or Self-Organizing Maps).

Compare clusters to approved-but-successful borrowers using similarity metrics (e.g. Mahalanobis distance, cosine similarity).

Score rejected applicants based on proximity to low-risk approved clusters.

Forecast risk using your existing supervised model on synthetic approvals from the top-scoring rejected group.

Validate by showing that your selected group has similar distributions in income, DTI, credit history, etc. to low-risk approved borrowers.

And some more datasets to use:
https://www.geeksforgeeks.org/machine-learning/loan-approval-prediction-using-machine-learning/
https://gigasheet.com/sample-data/credit-risk-dataset
https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data/data
And finnaly this link bellow has all of the data possible. 
https://www.listendata.com/2019/08/datasets-for-credit-risk-modeling.html



