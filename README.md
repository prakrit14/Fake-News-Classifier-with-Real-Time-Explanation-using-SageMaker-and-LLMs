# Fake News Classifier with Real-Time Explanation using SageMaker and LLMs

## Overview

This project presents a scalable, real-time fake news detection and explanation system, leveraging big data engineering and AI. It is developed as a **proof-of-concept (PoC)** demonstrating how misinformation detection can be scaled using cloud-based tools and modern AI interpretability frameworks.

> **This project repository is created in partial fulfillment of the requirements for the Big Data Analytics course offered by the Master of Science in Business Analytics program at the Carlson School of Management, University of Minnesota.**

We simulate a scalable end-to-end pipeline that includes:
- **Data ingestion and preprocessing** using Apache Spark
- **Model training and hosting** on AWS SageMaker (using a BERT-based classifier)
- **Real-time explanations** using the DeepSeek LLM API
- **Visualization and interactive demo** via Streamlit

The problem we address:  
> Over 5 million news articles are published online daily. Manual verification is infeasible — but automated, explainable verification is within reach.

---

## Repository Structure

- `app.py`  
  Flask app for deploying a lightweight API to interact with the fake news model and GenAI summarization.
  
- `distilbert_model_training_1.ipynb`  
  Jupyter notebook for fine-tuning a DistilBERT model on the fake/real news dataset for classification.

- `sagemaker.ipynb`  
  Notebook demonstrating the deployment of the trained model onto AWS SageMaker for scalable inference.

- `flier.pdf`  
  Project flier summarizing business value, technical approach, results, and key takeaways.


---

## Dataset

We use the following public dataset for model training and evaluation:

- [Kaggle Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

---

## Setup and Usage Instructions

Follow these steps to set up and run the project:

1. **Prepare AWS Environment**
   - Create an S3 bucket in AWS to store the model `.tar.gz` file.
   - Upload your model tar file to the S3 bucket.

2. **Launch a SageMaker Notebook**
   - Spin up a SageMaker Notebook Instance (select an instance like `ml.t2.medium`).
   - Clone this repository or upload the `sagemaker.ipynb` file.
   - Run the notebook to train your model and deploy it as a SageMaker endpoint.
   - Copy the SageMaker endpoint URL for later use.

3. **Local Application Setup**
   - Clone this GitHub repository locally.
   - Install dependencies using:
     ```bash
     pip install -r requirements.txt
     ```
   - Open `app.py` and **replace the placeholder** endpoint URL with your SageMaker endpoint URL.

4. **Run the Streamlit App**
   - Start the app by running:
     ```bash
     streamlit run app.py
     ```
   - The web app will allow users to submit article headlines/text and receive:
     - Real-time classification (Fake or Real)
     - Natural language explanation powered by a large language model (DeepSeek API)

---

## Key Technologies Used

- **Apache Spark**: Data preprocessing and feature engineering at scale
- **AWS SageMaker**: Model training, hosting, and inference
- **BERT**: Pre-trained transformer-based model for text classification
- **DeepSeek LLM API**: Explainability layer to interpret model predictions
- **Streamlit**: Interactive dashboard and user interface

---

## Key Limitations

- The dataset used is slightly outdated, which may introduce bias in model predictions.
- For production deployment, continuous live data ingestion and retraining pipelines are recommended.

---

## Additional Resources

- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html)
- [Apache Spark Documentation](https://spark.apache.org/docs/latest/)
- [DeepSeek API Documentation](https://deepseek.com)
- [Streamlit Documentation](https://docs.streamlit.io)

---

**Thank you for visiting our project!**  
