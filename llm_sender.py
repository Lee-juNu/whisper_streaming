def send_to_dify(conversation: str):
    """
    Send the conversation string to Dify workflow/run endpoint (blocking mode).
    """
    import os
    import httpx
    import json
    import logging

    logger = logging.getLogger("llm_sender")

    api_key = os.getenv("DIFY_API_CONVERSATION_WORKFLOW_API_KEY")
    api_url = os.getenv("DIFY_API_BASE_URL", "https://api.dify.ai/v1")

    if not api_key:
        logger.error("DIFY API key not set; skipping conversation send")
        return None

    payload = {
        "inputs": {"conversation": str(conversation)},
        "response_mode": "blocking",
        "user": "user-xyz",
    }
    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(
                f"{api_url}/workflows/run",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                json=payload,
            )
            data = resp.json()
            logger.debug(f"Dify conversation send response: {json.dumps(data)}")
            if resp.status_code >= 400:
                logger.error("Dify conversation send failed: %s", data)
                return None
            outputs = data.get("text") or data.get("data", {}).get("text")
            if isinstance(outputs, str):
                return outputs
            # Also check for outputs in 'outputs' key
            outputs_dict = data.get("outputs") or data.get("data", {}).get("outputs")
            if isinstance(outputs_dict, dict) and "text" in outputs_dict:
                return outputs_dict["text"]
            else:
                return None
    except Exception as http_err:
        logger.error("HTTP error during Dify conversation send: %s", str(http_err))
        return None
