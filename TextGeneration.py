import json
import boto3
import os
import logging
from botocore.exceptions import ClientError

LOG = logging.getLogger()
LOG.setLevel(logging.INFO)

region_name = os.getenv("region", "us-east-1")
s3_bucket = os.getenv("bucket")
model_id = os.getenv("model_id", "anthropic.claude-v2")

# Bedrock client used to interact with APIs around models
bedrock = boto3.client(service_name="bedrock", region_name=region_name)

# Bedrock Runtime client used to invoke and question the models
bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name=region_name)


def get_session_attributes(intent_request):
    session_state = intent_request["sessionState"]
    if "sessionAttributes" in session_state:
        return session_state["sessionAttributes"]

    return {}


def close(intent_request, session_attributes, fulfillment_state, message):
    intent_request["sessionState"]["intent"]["state"] = fulfillment_state
    return {
        "sessionState": {
            "sessionAttributes": session_attributes,
            "dialogAction": {"type": "Close"},
            "intent": intent_request["sessionState"]["intent"],
        },
        "messages": [message],
        "sessionId": intent_request["sessionId"],
        "requestAttributes": intent_request["requestAttributes"]
        if "requestAttributes" in intent_request
        else None,
    }


def lambda_handler(event, context):
    LOG.info(f"Event is {event}")
    accept = "application/json"
    content_type = "application/json"
    prompt = event["inputTranscript"]

    try:
        request = json.dumps(
            {
                "prompt": "\n\nHuman:" + prompt + "\n\nAssistant:",
                "max_tokens_to_sample": 4096,
                "temperature": 0.5,
                "top_k": 250,
                "top_p": 1,
                "stop_sequences": ["\\n\\nHuman:"],
            }
        )

        response = bedrock_runtime.invoke_model(
            body=request,
            modelId=model_id,
            accept=accept,
            contentType=content_type,
        )

        response_body = json.loads(response.get("body").read())
        LOG.info(f"Response body: {response_body}")
        response_message = {
            "contentType": "PlainText",
            "content": response_body["completion"],
        }
        session_attributes = get_session_attributes(event)
        fulfillment_state = "Fulfilled"

        return close(event, session_attributes, fulfillment_state, response_message)

    except ClientError as e:
        LOG.error(f"Exception raised while execution and the error is {e}")
