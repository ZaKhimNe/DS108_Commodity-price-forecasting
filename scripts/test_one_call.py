"""Test single LLM call with actual coffee data prompt."""
import requests, os, json
from dotenv import load_dotenv
load_dotenv()

auth_token = os.getenv("ANTHROPIC_AUTH_TOKEN", "")
api_key    = os.getenv("ANTHROPIC_API_KEY", "")
base_url   = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

with open("config/llm_farming_config.json") as f:
    model = json.load(f)["model"]

if auth_token:
    headers_auth = {"Authorization": f"Bearer {auth_token}"}
    print(f"Auth: Bearer (proxy) | endpoint: {base_url}")
else:
    headers_auth = {"x-api-key": api_key}
    print(f"Auth: x-api-key (official) | endpoint: {base_url}")
print(f"Model: {model}")

prompt = (
    "Brazil Coffee 2023: Production 54.94M bags, Arabica 38.2M (70%), "
    "Robusta 16.7M (30%), YoY change -2.5%, exports 41.2M bags, "
    "ending stocks 1.2M bags, stock-to-use 1.9%.\n"
    'Return ONLY JSON with 2 fields: {"signal":"one_of_the_enums","signal_reasoning":"max 20 words"}\n'
    "signal must be one of: bumper_crop_bullish, above_average, on_track, below_average, crop_stress_bearish, severe_stress_very_bearish"
)

r = requests.post(
    f"{base_url}/v1/messages",
    headers={**headers_auth, "anthropic-version": "2023-06-01",
             "content-type": "application/json"},
    json={"model": model, "max_tokens": 80,
          "messages": [{"role": "user", "content": prompt}]},
    timeout=30,
)

print(f"Status: {r.status_code}")
print(f"Body:   {r.text[:400]}")
if r.status_code == 200 and r.text.strip():
    try:
        text = r.json()["content"][0]["text"]
        print(f"LLM:    {text}")
        parsed = json.loads(text)
        print(f"Signal: {parsed['signal']}")
        print(f"Reason: {parsed['signal_reasoning']}")
    except Exception as e:
        print(f"Parse error: {e}")
