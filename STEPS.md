# AB_11 — Exact Setup and Run Steps

## 1. Open the project folder

```powershell
cd "C:\path\to\AB_11"
```

## 2. Create and activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

For Command Prompt instead of PowerShell:

```cmd
venv\Scripts\activate
```

## 3. Install requirements

The `-r` flag is required:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 4. Create `.env`

```powershell
copy .env.example .env
notepad .env
```

Set the OpenAI key. Set Supabase values only when running Option B.

## 5. Validate chunking without API calls

```powershell
python test_chunking.py
```

Expected:

```text
26 traces validated
104 chunks validated
```

## 6. Run Option A

```powershell
python evaluate_option_a.py
```

Verify:

```text
outputs_A/final_labels.csv                 26 records
outputs_A/final_labels.jsonl               26 records
outputs_A/chunk_evaluations.jsonl         104 records
outputs_A/chunks_for_embeddings.jsonl     104 records
```

## 7. Optional Option B: Supabase

Open:

```text
sql/01_create_vector_store.sql
```

Copy the entire file into a new Supabase SQL Editor query and click **Run**.

Then:

```powershell
python b1_create_embeddings_upsert.py
python b1_retrieve_example.py
```

## 8. Git

First verify `.env` is not tracked:

```powershell
git status
git ls-files .env
```

The second command must print nothing.

Then:

```powershell
git init
git branch -M main
git add .
git status
git commit -m "AB_11: verified chunk evaluation and semantic retrieval workflow"
git remote add origin https://github.com/MarcelaGreek/traces1M_AB.git
git remote -v
git push -u origin main
```

When replacing an existing remote intentionally:

```powershell
git fetch origin
git push --force-with-lease -u origin main
```
