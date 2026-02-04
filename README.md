# Project Summary

This project explores loan default risk using historical LendingClub data, with the goal of estimating probability of default at the time a loan is issued.

Early on, I realized that strong model performance can be misleading in this domain, so the project shifted from maximizing accuracy to understanding what signals are real, what signals are artificial, and what the model can actually be trusted to say.

The final outcome is a calibrated probability-of-default model that ranks risk meaningfully and produces interpretable estimates, rather than overstated predictions.

# Project Life
## Phase One — Initial Modeling and Suspicious Results

I started by building standard classification models to predict whether a loan would default.

The initial results looked very strong, but that raised concerns. After digging into the data and model behavior, I found that much of this performance came from:

forward-looking variables that wouldn’t exist at loan issuance,

random train-test splits that ignored time,

and survivorship bias in LendingClub’s approved-loan data.

At this point, I concluded that the early models were not valid for real-world decision-making, even though the metrics looked good.

## Phase Two — Correcting the Data and Framing

In the second phase, I rebuilt the project around realism rather than performance.

I removed all post-origination features and restricted the model to information that would be available at loan creation. I also switched to time-based validation, training on earlier years and testing on later years.

As expected, performance dropped. I treated this as a positive result, because it showed how much earlier performance had depended on information leakage.

## Phase Three — Moving From Classification to Risk Estimation

Instead of treating default as a binary outcome, I reframed the task as estimating probability of default.

This better reflects how risk models are used in practice. The goal isn’t to say whether a loan will default, but to quantify how risky it is relative to others.

I trained a CatBoost model and evaluated it using:

AUC for ranking ability,

PR-AUC to account for class imbalance,

and Brier score to assess probability accuracy.

The model showed meaningful ranking power (AUC ~0.69), but the raw probability outputs were clearly miscalibrated.

## Phase Four — Calibration, Stress Testing, and Interpretation

To address this, I applied probability calibration methods (Platt scaling and isotonic regression).

Calibration significantly improved the Brier score and aligned predicted probabilities with observed default rates across risk deciles, without materially changing ranking performance.

I also examined confusion matrices at different probability thresholds to understand tradeoffs between recall and precision. For example, at a conservative threshold, the model captures nearly all defaults but flags many safe loans as risky.

This reinforced that threshold selection is a policy choice, not a model flaw.

# Key Takeaways

Strong early performance often indicates data leakage rather than true predictive power.

Time-aware validation is critical for credit risk modeling.

Ranking performance and probability calibration matter more than raw accuracy.

After calibration, the model produces interpretable risk estimates that align with observed outcomes.

The model is suitable for risk ranking and decision support, not deterministic prediction.

This project taught me how easily models can look impressive for the wrong reasons, and how much work it takes to build something you can actually trust.


# ROC and TruePositive vs FalsePositive appitite 

I wondered, how do you choose whether you were risk avoident or risk seeking when it came to an ROC curve and found the below methods. 
Method 1: Youden’s J statistic (purely statistical)

Maximize: 𝐽 = TPR − FPR
          J=TPR−FPR

This finds the point farthest from the diagonal.

Pros
  Simple
  Common in medical testing
  Cons
  Assumes equal cost of false positives and false negatives
  Rarely appropriate for business or risk

## Exploratory Documentation
The remainder of this README documents the full analytical process,
including approaches that worked, failed, and informed later decisions.

#Data & Problem Framing

#Modeling Approach

#Evaluation & Stress Testing

#Limitations & Risks

## Development Notes (Exploratory Narrative)


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
![image](https://github.com/user-attachments/assets/7c162878-a61e-432c-bd9e-94f5b1469f44)
Also the grade in relation to default.
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

I started to compare the shared variables by grade type. To see if there was any substantial difference between them and got these results:
![image](https://github.com/user-attachments/assets/e88c9124-00ff-42f9-a28e-abf6f7986ca2)
![image](https://github.com/user-attachments/assets/3b7866a5-1339-4ae0-a00b-779b03fd50fc)
![image](https://github.com/user-attachments/assets/cd2dd4c8-c14d-449c-a6f5-b8c618b8b712)

I noticed that there was not much of a difference and combined the variables into a pca analysis comparing rejected loans to only the highest grade loans. 
![image](https://github.com/user-attachments/assets/e2a22c51-1c04-42eb-8de7-d94e045a0e3f)

There was no difference in distributions So, in conclusion, based on the three variables (debt-to-income-ratio, employee_length, loan_amount) that the rejected and approved data have in common. There appears to be no significant difference between the distributions meaning either of two things 1: the three variables do not explain enough of the variance that resulted in the applicants being sent to the rejected pile. 2: the applicants were unfairly sent to the reject pile.

I figure that it is the former. I thought about ranking the rejected file based on who is least risky but that is already a variable. The problem is that there is simply too much missing information from the dataset to say that they should be saved. After looking more into the data I realized that there could be a lot of reasons why they were rejected such as: they have previous defaults, a bad fico, ect. Given the information I simply can not even suggest that a person be allowed a loan. But I want to keep going with this idea. If I can find a dataset with rejected loans that has more information. I can use the structure that we built and try to save them from rejection.

So I pivoted the project instead take loans that were ready to be approved and built a machine learning model to determine if they would default. 
Using GridSearchCV with Pipeline for Lasso Regression I found that the full original cleaned dataset contained features that were past the point of my study, meaning that they told the model information about ongoing loans, when I needed the model to work with loans that would only be present at the time of creation. 
Those features were: Total principal repaid by the borrower up to this point. This excludes interest payments and reflects how much of the loan’s original amount has been returned.
Funds recovered after a loan has defaulted, often via collections or charged-off asset reclamation. A nonzero value here indicates a default occurred, and some money was clawed back.
Amount paid in the last payment cycle—includes both principal and interest. It’s useful to detect repayment patterns, stalling behavior, or payment shocks.
The upper bound of the borrower’s last reported FICO score range. LendingClub records FICO scores in ranges, and this represents the “optimistic” end. A falling range could signal distress.
I created a list of features that were not future dependent and filtered the cleaned dataset to only contain those features, resulting in 12 removed features. 
The mean-squared-error changed from a healthy .0960 to a not so great 0.229
🧠 Why It Got Worse
I removed forward-looking features (like payment history, FICO updates, etc.)—which is methodologically correct when trying to simulate real-time underwriting at loan issuance. But it also removes valuable predictive power, so the MSE naturally worsens. 
0.229 MSE suggests the model’s predictions are, on average, off by about √0.229 ≈ 0.48 units, which isn't terrible depending on the label encoding—but in classification (especially if using 0/1 as labels), this likely indicates lots of uncertainty around predicting 1s (defaults).

Some more datasets to use:
https://www.geeksforgeeks.org/machine-learning/loan-approval-prediction-using-machine-learning/
https://gigasheet.com/sample-data/credit-risk-dataset
https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data/data
And finnaly this link bellow has all of the data possible. 
https://www.listendata.com/2019/08/datasets-for-credit-risk-modeling.html

So I continued with the project but changed my scope after listening to how defauting loans can actually be very bad for the consumer. That was much more strightforward and I trained the data on some ML models 
<img width="993" height="364" alt="image" src="https://github.com/user-attachments/assets/01ecc41d-6606-493d-bb4c-facaf625db2b" />
From thsoe I chose the best performing models
for Knn 
<img width="417" height="236" alt="image" src="https://github.com/user-attachments/assets/b7e0b041-d6e4-4cb0-8321-53468c0420e2" />
for Random Forest
<img width="419" height="231" alt="image" src="https://github.com/user-attachments/assets/b3ba0fda-287c-4f02-af1e-b59ab4d23591" />
For XGBoost 
<img width="432" height="228" alt="image" src="https://github.com/user-attachments/assets/5beec5b8-5219-4afc-a32f-79eeb1ccb453" />

I saved those into a python file that launches and API and uses these models as a evaluation tool on any one loan with the right inputs and take the best guess of the three models as to if the loan will default. 

# Part Two

I will finish the project and turn the result into a powerpoint. 
When we left off I had accuracy, precision, and recall suspiciously high 
I want to evaluate the predictive power of my model 

Initial evaluation metrics were unexpectedly high. The following stress tests were conducted to assess leakage, overfitting, and robustness.

I learned about the CatBoost which is a gradient descent 


