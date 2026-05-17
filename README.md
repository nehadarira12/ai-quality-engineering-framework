# AI Quality Engineering Framework

A practical Python framework for testing AI systems across four 
critical layers — built from real experience testing a 
patient-facing Voice AI platform in a US healthcare environment.

## Why This Exists

AI is being embedded into products that affect real people.

A broken button is a UX issue.
A broken AI guardrail is a safety or compliance failure.

Most QA frameworks test what a system does.
This framework also tests what a system should never do.

## What This Framework Tests

### 1. Data Validation

Clean data is the foundation of reliable AI. If the data going 
into an AI model is wrong, every output will be wrong.

Tests cover:
- Field completeness and required field validation
- Data type correctness
- Sensitivity classification
- PII detection in definitions and descriptions
- Naming convention consistency
- Duplicate detection

### 2. LLM Output Evaluation

AI responses must be relevant, accurate, and grounded in context.

Tests cover:
- Answer relevancy — did the AI answer the actual question?
- Hallucination detection — did the AI invent facts not in the context?

Real scenario: A Voice AI told a patient their copay was $25 when 
no copay information existed in the system. Our hallucination test 
catches this before it reaches a real patient.

### 3. Semantic Model Validation

Semantic models define what business data means in context.
A wrong definition silently corrupts every report and insight 
built on top of it.

Tests cover:
- Business definition accuracy
- Metric formula validation
- Hallucination in agent-generated definitions
- Internal consistency between entity names and metrics

### 4. Agent Output Validation

AI agents generate data catalogs, metric definitions, and business 
glossaries. These tests ensure agent outputs are complete and 
accurate before they are published to downstream systems.

Tests cover:
- Catalog entry completeness
- Metric definition structure
- Agent goal completion rate
- Column hallucination detection

### 5. Adversarial Testing

AI systems that interact with real people must be resilient
against manipulation, deception, and security attacks.

Tests cover:
- Prompt injection resistance
- Role confusion attempts
- Authority escalation attempts
- Urgency and emotional manipulation
- Soft re-identification attempts
- Memory leakage between sessions
- PII exposure in error messages

These scenarios map directly to OWASP LLM Top 10 risks.

## Project Structure
```
ai-quality-engineering-framework/
│
├── conftest.py                       # Shared GroqModel setup
├── requirements.txt                  # All dependencies
│
├── data_validation/
│   └── test_data_quality.py          # Completeness, types, sensitivity, PII
│
├── llm_evaluation/
│   └── test_llm_evaluation.py        # Relevancy and hallucination detection
│
├── semantic_model_validation/
│   └── test_semantic_accuracy.py     # Business definition accuracy
│
├── agent_output_validation/
│   └── test_agent_outputs.py         # Catalog entries and metric validation
│
└── adversarial_testing/
    └── test_adversarial.py           # Prompt injection, manipulation, leakage
```

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| DeepEval | LLM evaluation framework |
| Groq | Free LLM judge (LLaMA 3.3 70B) |
| Pytest | Test runner |
| GitHub Actions | CI runs tests on every push |

## How to Run

Clone the repo

```bash
git clone https://github.com/nehadarira12/ai-quality-engineering-framework.git
cd ai-quality-engineering-framework
```

Install dependencies

```bash
pip install -r requirements.txt
```

Set your Groq API key

```bash
export GROQ_API_KEY="your-groq-key-here"
```

Get a free key at console.groq.com

Run all tests

```bash
pytest --tb=short -v
```

Run a specific module

```bash
pytest data_validation/ -v
pytest llm_evaluation/ -v
pytest adversarial_testing/ -v
```

## Key Concepts

**What is Hallucination Testing?**
Hallucination is when an AI invents facts that do not exist
in the provided context. In healthcare, a hallucinated copay
amount could cause a patient to make a wrong financial decision.
In data products, a hallucinated formula corrupts every metric
built on top of it.

**What is Agentic Testing?**
Agentic AI takes autonomous actions across multiple steps
to complete a goal. Testing it means validating not just
the final output but the reasoning process and decision
quality at every step.

**What is Adversarial Testing?**
Adversarial testing validates that an AI system maintains
its guardrails when users attempt to manipulate, deceive,
or exploit it. This is not optional for AI systems that
interact with real people in regulated environments.

## Background

Built from real experience leading QA for a patient-facing
Voice AI platform in a US healthcare environment, where AI
quality is not optional it is a compliance requirement.

## Author

**Neha Kumari** — Senior QA Engineer | ISTQB Certified

[LinkedIn](https://www.linkedin.com/in/neha-kumari-qa/) | [GitHub](https://github.com/nehadarira12)
