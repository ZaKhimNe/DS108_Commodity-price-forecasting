"""Quick test: find which model works on the configured proxy."""
import requests, os, json
from dotenv import load_dotenv

load_dotenv()
auth_token = os.getenv("ANTHROPIC_AUTH_TOKEN", "")
api_key    = os.getenv("ANTHROPIC_API_KEY", "")
base_url   = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

# Proxy (1gw.gwai.cloud) dùng Bearer token; Official Anthropic dùng x-api-key
if auth_token:
    headers_auth = {"Authorization": f"Bearer {auth_token}"}
    key_display  = auth_token[:15]
else:
    headers_auth = {"x-api-key": api_key}
    key_display  = api_key[:15]

print(f"Endpoint: {base_url}")
print(f"Auth:     {'Bearer' if auth_token else 'x-api-key'} | key: {key_display}...\n")

MODELS = [
    # Claude 4
    "claude-opus-4-5",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
    # Claude 3.7 / 3.5
    "claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    # Claude 3
    "claude-3-haiku-20240307",
    "claude-3-sonnet-20240229",
    "claude-3-opus-20240229",
    # Claude 2
    "claude-2.1",
    "claude-instant-1.2",
    # AWS Bedrock style IDs
    "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    "us.anthropic.claude-3-haiku-20240307-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
]

found = []
for m in MODELS:
    try:
        r = requests.post(
            f"{base_url}/v1/messages",
            headers={**headers_auth, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": m, "max_tokens": 10,
                  "messages": [{"role": "user", "content": "hi"}]},
            timeout=10,
        )
        if r.status_code == 200:
            print(f"  OK  {m}")
            found.append(m)
        else:
            body = r.text[:80] if r.text else "(empty)"
            print(f"  {r.status_code} | {m} | {body}")
    except Exception as e:
        print(f"  ERR {m} | {e}")

print()
if found:
    print("Working models:", found)
    # Auto-update config
    with open("config/llm_farming_config.json") as f:
        cfg = json.load(f)
    cfg["model"] = found[0]
    with open("config/llm_farming_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Config updated → model: {found[0]}")
else:
    print("No working models found. Check account/subscription at:", base_url)
