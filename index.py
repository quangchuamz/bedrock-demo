import json
import os
import boto3
import streamlit as st
import time
import requests
import dotenv

dotenv.load_dotenv()
session = boto3.Session()
bedrock = session.client(
    service_name="bedrock-runtime",
    region_name=os.getenv("AWS_DEFAULT_REGION"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)


## LOGIC ##
def get_ms() -> int:
    return round(time.time() * 1000)


def get_aws_ai_streaming(prompt, max_tokens, temperature):
    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        }
    )

    index, firstTime, totalTime = 0, 0, 0
    response_output_tokens = ""
    start = get_ms()
    response = bedrock.invoke_model_with_response_stream(modelId="anthropic.claude-3-haiku-20240307-v1:0", body=body)  # invoke the streaming method

    for event in response.get("body"):
        print(event)
        chunk = json.loads(event["chunk"]["bytes"])
        if chunk["type"] == "content_block_delta":
            if chunk["delta"]["type"] == "text_delta":
                yield chunk["delta"]["text"]

        if chunk["type"] == "message_stop":
            response_output_tokens = chunk["amazon-bedrock-invocationMetrics"]["outputTokenCount"]

        if index == 0:
            firstTime = get_ms() - start

        index += 1

    totalTime = get_ms() - start
    st.markdown(f"Chunk Response: {firstTime}ms<br />Total Response: {totalTime}ms<br />Output Tokens: {response_output_tokens}", unsafe_allow_html=True)


def get_aws_ai(prompt, max_tokens, temperature):
    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        }
    )
    start = get_ms()
    response = bedrock.invoke_model(modelId="anthropic.claude-3-haiku-20240307-v1:0", body=body)
    response_body = json.loads(response.get("body").read())
    response_text = response_body.get("content")[0].get("text")
    response_output_tokens = response_body.get("usage").get("output_tokens")
    response_time = get_ms() - start
    return (response_text, response_output_tokens, response_time)


def get_azure_ai(prompt, max_tokens, temperature):
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": 0,
    }
    headers = {"Content-Type": "application/json; charset=utf-8", "api-key": os.getenv("AZURE_API_KEY")}
    start = get_ms()
    response = requests.post("https://dscai.openai.azure.com/openai/deployments/test-gpt-35-16k/chat/completions?api-version=2023-05-15", json=body, headers=headers)
    response_body = response.json()
    response_text = response_body.get("choices")[0].get("message").get("content")
    response_output_tokens = response_body.get("usage").get("total_tokens")
    response_time = get_ms() - start
    return (response_text, response_output_tokens, response_time)


## WEB_UI ##
st.set_page_config(page_title="Chatbot")  # HTML title
st.title("Chatbot")  # page title

max_tokens = st.number_input("Max Tokens", value=10000)
temperature = st.number_input("Temperature", value=0.0)
prompt = st.text_area("Input text")
go_button = st.button("Go", type="primary")

if go_button:  # code in this if block will be run when the button is clicked
    with st.spinner("Running..."):
        st.subheader("AWS STREAMING")
        st.write_stream(get_aws_ai_streaming(prompt=prompt, max_tokens=max_tokens, temperature=temperature))

        aws = get_aws_ai(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
        azure = get_azure_ai(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("AWS")
            st.write(aws[0])
            st.markdown(f"Response: {aws[2]}ms<br />Output Tokens: {aws[1]}", unsafe_allow_html=True)
        with col2:
            st.subheader("AZURE")
            st.write(azure[0])
            st.markdown(f"Response: {azure[2]}ms<br />Output Tokens: {azure[1]}", unsafe_allow_html=True)
