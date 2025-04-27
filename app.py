import streamlit as st
import boto3
import json
import requests

#sagemaker config
ENDPOINT_NAME = "<Enter Sagemaker Endpoint Name Here>"  
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


# call LLM for explainable component
def explainabilityCall(article_text: str, prediction: dict) -> dict:

    openrouter_api_key = "<Insert Openrouter DeepSeek API Key Here>"
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
    else:
        print(response.status_code)
        return {'response_error': response.status_code}
            

# Sample articles stored as strings
def load_sample_articles():
    return {
        "Florida Man Arrested for Attempting to Run to London in a Hamster Wheel": (
            "In 2023, a Florida man was rescued by the U.S. Coast Guard while attempting to cross the Atlantic Ocean "+
            "in a homemade 'hamster wheel' device. The man, who had attempted similar stunts before, claimed he "+
            "was trying to reach London. Authorities deemed the vessel unsafe and took him into custody for his "+
            "own protection."
            "(Source: BBC News, USA Today)"
        ),
        "Government to Ban Rain on Weekends": (
            "A leaked memo from top government officials reveals plans to control the weather and ban rain on"
            "weekends by 2026. Using 'cloud seeding' technologies, the government reportedly aims to ensure sunny "+
            "Saturdays and Sundays across the nation."
            
        ),
        "Miracle Cure Found for Baldness": (
            "Scientists at a small university claim to have discovered a miracle cure for baldness that requires only drinking two glasses of pineapple juice each day."+
            "According to the report, 100% of participants grew a full head of hair within a week,"+ 
            "though no clinical trials have been conducted"
        ),
    }


# app UI
def main():
    st.title("📰 Fake News Detector")
    st.write("Enter a news headline and body to check if it's likely real or fake.")
    
        # Load samples
    samples = load_sample_articles()
    options = ["-- Select an example --"] + list(samples.keys())
    choice = st.selectbox("Example Articles", options)

    # Prepare text area default value
    default_text = samples.get(choice, "")
    article_text = st.text_area(
        "Article Text", value=default_text, height=200
    )


    # headline = st.text_input("Headline")
    # body = st.text_area("Article Body", height=200)

    if st.button("Analyze"):
        if not article_text.strip():
            st.warning("Please select a sample headline or paste an article to analyze.")
            return

        # combine inputs like during training
        full_text = f"{choice.strip()} {article_text.strip()}"

        # run 
        try:
            prediction = makePrediction(full_text)
        
        except Exception as e:
            print(f"Something went wrong: {e}")
            st.write(f"Something went wrong: {e}")
            
            prediction = {'label': 'real', 'prob': 0.51}
            st.write(f'proceeding with example prediction: {prediction}')
            
        try:   
            explanation = explainabilityCall(full_text, prediction)
            
        except Exception as e:
            print(f"Something went wrong: {e}")
            st.write(f"Something went wrong: {e}")
            
            
                

        # results
        st.subheader("Prediction")
        st.write(f"**Label:** {prediction['label']}  ")
        st.write(f"**Probability:** {prediction['prob']}  ")

        st.subheader("Explainability")
        if 'response_error' not in explanation.keys():
            st.write(f"**Model & LLM:** {explanation.get('agreeOrNot', 'Unavailable')}  ")
            st.write(f"**Explanation:** {explanation.get('explanation', 'No explanation returned.')}  ")
        else:
            print(f"LLM API response issue: response: {explanation['response_error']}")
            st.write(f"LLM API response issue: response: {explanation['response_error']}")

if __name__ == "__main__":
    main()
