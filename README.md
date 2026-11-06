# ML_Airline_Passenger_Satisfaction

Build and Evaluate ML algorithms like Logistic Regression, Decision Tree, KNN, Naive Bayes and SVM on Airlines passenger satisfaction dataset.
Based on the **Airline Passenger Satisfaction** dataset on Kaggle, here is a comprehensive overview of the dataset description and feature details:

---

## **a. Problem Statement**

* **Context**: The dataset contains a customer satisfaction survey from an airline, capturing various passenger demographics, travel preferences, and specific ratings of in-flight and ground services.
* **Objective**: The primary goal is to build classification models that can accurately predict whether a passenger is **satisfied** or **neutral/dissatisfied** based on their survey responses and flight details.
* **Data Split**: The dataset is typically pre-split into a training set (approx. 103,904 rows) and a test set (approx. 25,976 rows).

Source: https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction

---

## **b. Dataset description**

The dataset consists of 25 columns (including index/ID columns that are typically dropped during preprocessing), which can be categorized into four main groups:

### **1. Passenger Demographics & Information**

* **`id` / `Unnamed: 0**`: Unique row identifiers (usually dropped during modeling). Note: For  training or testing it is removed.
* **`Gender`**: The gender of the passengers (`Female`, `Male`).
* **`Customer Type`**: Indicates customer loyalty status (`Loyal Customer`, `disloyal Customer`).
* **`Age`**: The actual numerical age of the passenger.

### **2. Travel Specifics**

* **`Type of Travel`**: The purpose of the passenger's flight (`Personal Travel`, `Business travel`).
* **`Class`**: The travel class onboard the plane (`Business`, `Eco`, `Eco Plus`).
* **`Flight Distance`**: The flight distance of the journey in miles.

### **3. Service Satisfaction Ratings (Rated on a scale from 0 to 5, where 0 represents "Not Applicable" / unrated)**

* **`Inflight wifi service`**: Satisfaction level with the onboard Wi-Fi service.
* **`Departure/Arrival time convenient`**: Convenience rating of departure and arrival times.
* **`Ease of Online booking`**: Satisfaction level with the online booking process.
* **`Gate location`**: Satisfaction level with airport gate locations.
* **`Food and drink`**: Satisfaction level with food and beverages served.
* **`Online boarding`**: Satisfaction level with the online boarding experience.
* **`Seat comfort`**: Satisfaction level with seat comfort.
* **`Inflight entertainment`**: Satisfaction level with onboard entertainment systems.
* **`On-board service`**: Satisfaction level with crew/on-board service.
* **`Leg room service`**: Satisfaction level with leg room space.
* **`Baggage handling`**: Satisfaction level with baggage handling.
* **`Checkin service`**: Satisfaction level with check-in service.
* **`Inflight service`**: Satisfaction level with general inflight service.
* **`Cleanliness`**: Satisfaction level with cabin cleanliness.

### **4. Flight Delay Information**

* **`Departure Delay in Minutes`**: Number of minutes delayed upon departure.
* **`Arrival Delay in Minutes`**: Number of minutes delayed upon arrival (often contains small amounts of missing values that require median imputation).

### **5. Target Variable**

* **`satisfaction`**: The overall passenger satisfaction level (`satisfied` vs. `neutral or dissatisfied`).

## **c. GitHub Repository Link**

## **d. Models used**
Comparison Table with the evaluation metrics calculated for all the 5 
models as below:  
|S.No|Model|Accuracy|AUC Score|Precision|Recall|F1 Score|MCC Score|
|-|-|-|-|-|-|-|-|
|1|Logistic Regression|0.8764|0.9281|0.8690|0.8418|0.8551|0.7477|
|2|Decision Tree|0.9439|0.9435|0.9307|0.9406|0.9356|0.8859|
|3|KNN|0.9280|0.9691|0.9464|0.8838|0.9140|0.8536|
|4|Naive Bayes|0.8614|0.9217|0.8562|0.8174|0.8364|0.7169|
|5|Support Vector Machine|0.9539|0.9892|0.9546|0.9384|0.9464|0.9061|

Observations details about the models metrics.

|ML Model Name|Observation about model performance|
|-|-|
|Logistic Regression|Achieved an accuracy of 87.64%. AUC score of 0.9281 indicates good ability to distinguish between classes. F1 score of 0.8551 indicates reasonably balanced precision and recall. Performance is comparatively lower than the other three leading models.Conclusion: Logistic Regression provides a good baseline but may not capture complex nonlinear relationships in the dataset effectively.|
|Decision Tree|Achieved an accuracy of 94.39%.AUC score of 0.9435 indicates good classification capability.F1 score of 0.9356 shows a strong balance between precision and recall.MCC score of 0.8859 indicates strong agreement between actual and predicted classifications.Conclusion: Decision Tree performs very well and captures patterns in the dataset effectively. However, it may be susceptible to overfitting.|
|KNN|Achieved an accuracy of 92.80%.Obtained an AUC score of 0.9691, indicating excellent class discrimination.Highest precision among the first four models at 0.9464.Recall of 0.8838 indicates that some actual positive cases may be missed.Conclusion: KNN demonstrates strong classification capability, particularly in identifying positive predictions accurately. However, its recall is lower than Decision Tree and SVM.|
|Naive Bayes|Achieved an accuracy of 86.14%, the lowest among the five models.AUC score of 0.9217 indicates reasonable class discrimination.F1 score of 0.8364 is also the lowest.Conclusion: Naive Bayes provides reasonable performance but may be less suitable for this dataset, possibly because its feature independence assumption does not fully represent the relationships among the features.|
|Support Vector Machine|Achieved the highest accuracy of 95.39%.Highest AUC score of 0.9892, indicating excellent class discrimination.Highest precision of 0.9546.Recall of 0.9384 indicates that it successfully identifies most positive cases.Highest F1 score of 0.9464.Highest MCC score of 0.9061.Conclusion: SVM demonstrates the strongest overall classification performance, achieving an excellent balance between precision and recall.|

### Final Conclusion

In this experiment, five machine learning classification algorithms—Logistic Regression, Decision Tree, K-Nearest Neighbors, Naive Bayes, and Support Vector Machine—were trained and evaluated on the given dataset.

The performance of these models was compared using Accuracy, AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

Among all five algorithms, **Support Vector Machine (SVM) emerged as the best-performing model**, achieving the highest accuracy of 95.39%, AUC score of 0.9892, F1 score of 0.9464, and MCC score of 0.9061.

Decision Tree also demonstrated strong performance with 94.39% accuracy and a 0.9356 F1 score, making it a competitive alternative. KNN achieved good precision and AUC scores, while Logistic Regression and Naive Bayes showed comparatively lower performance.

Overall, SVM is selected as the final winning model because it provides the best balance between classification accuracy, precision, recall, and overall predictive performance. Its high AUC score also indicates excellent class discrimination.

Therefore, based on the evaluated metrics, **SVM is the most suitable classification algorithm among the five models tested on this dataset.**
