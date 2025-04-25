import streamlit as st
import boto3
import json
import requests

#sagemaker config
ENDPOINT_NAME = "huggingface-pytorch-inference-2025-04-24-23-07-50-770"  
REGION = "us-east-1"                  

# real model prediction call
runtime = boto3.client("sagemaker-runtime", region_name=REGION)

def makePrediction(article_text: str) -> dict:
    payload = json.dumps({"inputs": article_text})
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=payload
    )
    result = json.loads(response["Body"].read().decode())


    if isinstance(result, list) and isinstance(result[0], dict) and "label" in result[0]:
        label = result[0]["label"]
        prob = result[0].get("score", 0.0)
    else:
        prob = result[0]
        label = "fake" if prob > 0.5 else "real"

    return {"label": label, "prob": round(prob, 3)}

# explainability placeholder 
def explainabilityCall(article_text: str, prediction: dict) -> dict:

    openrouter_api_key = "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"
    system_prompt = (
        "You are an expert AI assisting with fake news detection. "
        "You are provided with a news article title, its full content, and a prediction label (either 'Fake' or 'Real'). "
        "Your task is to independently analyze the article and decide whether you AGREE or DISAGREE with the prediction. "
        "You must respond ONLY with a **strictly valid JSON object**, with these two keys:\n\n"
        "1. agreeOrNot: A short sentence saying if you agree or not with the prediction\n"
        "2. explanation: A detailed explanation\n\n"
        "Do not add any extra text, commentary, or formatting. Return only JSON."
        "Respond ONLY in this JSON format:\n"
        f"""{{
                "agreeOrNot": "<Your agreement or disagreement with the prediction of the external model>",
                "explanation": "<Your detailed explanation for the task here>"
            }}"""
    )
    user_prompt = (
        f"Article: {article_text}\n\n"
        f"Prediction of the external model (as a dictionary of label and probability): {prediction}\n\n"
        f"Do you agree with this prediction? Provide your answer in JSON format only."
    )
    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://yourprojectname.com",
        "X-Title": "Fake News Checker"
    }
    payload = {
        "model": "deepseek/deepseek-r1:free",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.7
    }

    url = "https://openrouter.ai/api/v1/chat/completions"
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        try:
            data = response.json()
            message_content = data["choices"][0]["message"]["content"]

            # Safely parse the JSON string in the response
            parsed_result = json.loads(message_content)
            
            # Check for required keys
            if "agreeOrNot" in parsed_result and "explanation" in parsed_result:
                return parsed_result
            else:
                return {
                    "agreeOrNot": "Unavailable",
                    "explanation": "Missing expected keys in the response."
                }

        except Exception as e:
            return {
                "agreeOrNot": "Error",
                "explanation": f"Failed to parse DeepSeek response: {str(e)}"
            }
        else:
            return {
            "agreeOrNot": "Error",
            "explanation": f"API error {response.status_code}: {response.text}"
        }

# app UI
def main():
    st.title("📰 Fake News Detector")
    st.write("Enter a news headline and body to check if it's likely real or fake.")

    headline = st.text_input("Headline")
    body = st.text_area("Article Body", height=200)

    if st.button("Analyze"):
        if not headline.strip() and not body.strip():
            st.warning("Please enter a headline or body to analyze.")
            return

        # combine inputs like during training
        full_text = f"{headline.strip()} {body.strip()}"

        # run 
        prediction = makePrediction(full_text)
        explanation = explainabilityCall(full_text, prediction)

        # results
        st.subheader("Prediction")
        st.write(f"**Label:** {prediction['label']}  ")
        st.write(f"**Probability:** {prediction['prob']}  ")

        st.subheader("Explainability")
        st.write(f"**Model & LLM:** {explanation.get('agreeOrNot', 'Unavailable')}  ")
        st.write(f"**Explanation:** {explanation.get('explanation', 'No explanation returned.')}  ")

if __name__ == "__main__":
    main()
