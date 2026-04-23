# Medical QA Agent MVP

This project builds a Chinese-first medical QA agent for health education.

## Safety boundary

- Informational and educational use only.
- No personalized diagnosis, treatment plans, or prescription dosing.
- If high-risk symptoms are reported, direct users to in-person care or emergency services.

## Current status

- Step 1 complete: project scaffolding and dependencies are in place.
- Step 2 complete: configuration system and API client scaffold are ready.
- Step 3 ready to run: MedlinePlus and PubMed download scripts are available.

## Quick check

Run this command from the project root to verify configuration loading:

python scripts/smoke_config.py

## Step 3 data download

Run MedlinePlus download:

python scripts/download_medlineplus.py --max-records 500

Run PubMed download:

python scripts/download_pubmed.py --query "(hypertension[Title/Abstract]) OR (diabetes[Title/Abstract])" --max-results 200

## Step 4 corpus preparation

Merge MedlinePlus and PubMed into normalized docs plus chunk files:

python scripts/prepare_corpus.py

## Step 5 vector index and retrieval test

Build local FAISS index:

python scripts/build_index.py

If HuggingFace is unreachable, the builder will automatically fall back to a local hashing embedding backend, so indexing can still complete offline.

Run a retrieval smoke test:

python scripts/search_index.py --query "高血压和糖尿病的关系" --top-k 3

## Step 6 end-to-end QA

Run retrieval + answer generation (LLM if configured, fallback summary otherwise):

python scripts/ask_qa.py --question "高血压和糖尿病有什么关系？" --top-k 5 --style empathetic

## Step 7 Streamlit web app

Launch web UI:

streamlit run app/streamlit_app.py

Note: run this command from project root (C:/agent).

## Step 8 retrieval evaluation

Run retrieval evaluation (Recall@K + MRR):

python scripts/eval_retrieval.py --dataset eval/retrieval_eval_sample.jsonl --ks 1,3,5,10

## Step 9 LangGraph agent

This project now uses a LangGraph-first agent:

- `MessagesState` for messages
- `ToolNode` for tools
- checkpoint-based thread memory
- SQLite archive for history lookup
- optional summary memory via `MEMORY_LLM_*`

Run QA:

python scripts/ask_qa.py --question "高血压和糖尿病有什么关系？" --top-k 5

In Streamlit, you can change `用户ID`, `回答风格`, `检索条数`, and inspect or clear the current thread.

## Speed tuning (local Ollama)

- Lower generation length in .env: LLM_MAX_TOKENS=220~320
- Keep temperature low: LLM_TEMPERATURE=0.1
- Prefer smaller Top-K for daily use (e.g. 3)
- Use smaller model for latency-sensitive scenarios (e.g. qwen2.5:3b)

## External APIs (AMap + Aliyun SMS + Registration API)

This project can call external tools for hospital search, registration, and reminders.

Set these env vars in `.env`:

- `AMAP_API_KEY`: AMap Web Service key
- `AMAP_BASE_URL`: default `https://restapi.amap.com`
- `ALIYUN_ACCESS_KEY_ID` / `ALIYUN_ACCESS_KEY_SECRET`
- `ALIYUN_SMS_SIGN_NAME` / `ALIYUN_SMS_TEMPLATE_CODE`
- `ALIYUN_SMS_ENDPOINT`: default `dysmsapi.aliyuncs.com`
- `ALIYUN_SMS_REGION`: default `cn-hangzhou`
- `HOSPITAL_REG_API_BASE_URL` / `HOSPITAL_REG_API_KEY` (your own registration provider)

Current tool mapping:

- `find_hospital_guidance`: calls AMap POI text search
- `send_followup_reminder`: calls Aliyun SMS `SendSms`
- `schedule_followup_reminder`: creates scheduled reminder tasks in SQLite queue
- `schedule_registration`: posts appointment payload to external registration API

Recommended install for SMS tool:

`pip install aliyun-python-sdk-core`

Reminder dispatch (run this periodically by Task Scheduler / cron):

`python scripts/dispatch_reminders.py --max-tasks 20 --max-retry 3`
