from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
import os
import re
from typing import Dict, Any, Union, List


# ---------------- Initialize ----------------
app = FastAPI(title="LLM Model API", version="3.4")

# ✅ Load environment variables
load_dotenv()

# ✅ Fetch Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found. Please set it in your .env or environment variables.")

# ✅ Configure Gemini Client
genai.configure(api_key=GEMINI_API_KEY)
MODEL_ID = "gemini-2.5-flash"


# ---------------- Schema ----------------
class BiomarkerRequest(BaseModel):
    albumin: float = Field(default=3.2, description="Albumin level in g/dL")
    creatinine: float = Field(default=1.4, description="Creatinine level in mg/dL")
    glucose: float = Field(default=145, description="Glucose level in mg/dL")
    crp: float = Field(default=12.0, description="C-reactive protein in mg/L")
    mcv: float = Field(default=88, description="Mean corpuscular volume in fL")
    rdw: float = Field(default=15.5, description="Red cell distribution width in %")
    alp: float = Field(default=120, description="Alkaline phosphatase in U/L")
    wbc: float = Field(default=11.8, description="White blood cell count in ×10^3/μL")
    lymphocytes: float = Field(default=20, description="Lymphocyte percentage")
    hb: float = Field(default=13.0, description="Hemoglobin in g/dL")
    pv: float = Field(default=2.1, description="Plasma volume in L (converted internally if needed)")
    age: int = Field(default=52, description="Patient age in years")
    gender: str = Field(default="female", description="Gender of the patient")
    height: float = Field(default=165, description="Height in cm")
    weight: float = Field(default=70, description="Weight in kg")


# ---------------- Cleaning Utility ----------------
def clean_json(data: Union[Dict, List, str]) -> Union[Dict, List, str]:
    """Recursively removes separators, extra whitespace, and artifacts from all string values."""
    if isinstance(data, str):
        text = re.sub(r"-{3,}", "", data)
        text = re.sub(r"\s+", " ", text)
        text = text.strip(" -\n\t\r")
        return text
    elif isinstance(data, list):
        return [clean_json(i) for i in data if i and clean_json(i)]
    elif isinstance(data, dict):
        return {k.strip(): clean_json(v) for k, v in data.items()}
    return data


# ---------------- Parser ----------------
def parse_medical_report(text: str):
    """
    Parses Gemini markdown response → structured JSON.
    Detects section headers, **bold keys**, and table entries.
    """
    def clean_line(line: str) -> str:
        return re.sub(r"[\-\*\u2022]+\s*", "", line.strip())

    def parse_bold_entities(block: str) -> Dict[str, str]:
        """Extracts **bold** entities and maps text until next bold or section."""
        entities = {}
        pattern = re.compile(r"\*\*(.*?)\*\*(.*?)(?=\*\*|###|$)", re.S)
        for match in pattern.finditer(block):
            key = match.group(1).strip().strip(":")
            val = match.group(2).strip().replace("\n", " ")
            val = re.sub(r"\s+", " ", val)
            if key:
                entities[key] = val
        return entities

    data = {
        "executive_summary": {"top_priorities": [], "key_strengths": []},
        "system_analysis": {},
        "personalized_action_plan": {},
        "interaction_alerts": [],
        "normal_ranges": {},
        "biomarker_table": []
    }

    # --- Executive Summary ---
    exec_match = re.search(r"###\s*Executive Summary(.*?)(?=###|$)", text, re.S | re.I)
    if exec_match:
        block = exec_match.group(1)
        priorities = re.findall(r"\d+\.\s*(.*?)\n", block)
        if priorities:
            data["executive_summary"]["top_priorities"] = [clean_line(p) for p in priorities]
        strengths_match = re.search(r"\*\*Key Strengths:\*\*(.*)", block, re.S)
        if strengths_match:
            strengths_text = strengths_match.group(1)
            strengths = [clean_line(s) for s in strengths_text.splitlines() if clean_line(s)]
            data["executive_summary"]["key_strengths"] = strengths

    # --- System Analysis ---
    sys_match = re.search(r"###\s*System[- ]Specific Analysis(.*?)(?=###|$)", text, re.S | re.I)
    if sys_match:
        sys_block = sys_match.group(1)
        data["system_analysis"] = parse_bold_entities(sys_block)

    # --- Personalized Action Plan ---
    plan_match = re.search(r"###\s*Personalized Action Plan(.*?)(?=###|$)", text, re.S | re.I)
    if plan_match:
        plan_block = plan_match.group(1)
        data["personalized_action_plan"] = parse_bold_entities(plan_block)

    # --- Interaction Alerts ---
    alerts_match = re.search(r"###\s*Interaction Alerts(.*?)(?=###|$)", text, re.S | re.I)
    if alerts_match:
        alerts_block = alerts_match.group(1)
        alerts = [clean_line(a) for a in alerts_block.splitlines() if clean_line(a)]
        data["interaction_alerts"] = alerts

    # --- Normal Ranges ---
    normal_match = re.search(r"###\s*Normal Ranges(.*?)(?=###|$)", text, re.S | re.I)
    if normal_match:
        normal_block = normal_match.group(1)
        for match in re.findall(r"-\s*([^:]+):\s*([^\n]+)", normal_block):
            biomarker, rng = match
            data["normal_ranges"][biomarker.strip()] = rng.strip()

    # --- Tabular Mapping ---
    table_match = re.search(r"###\s*Tabular Mapping(.*)", text, re.S | re.I)
    if table_match:
        table_block = table_match.group(1)
        # robust row matcher: capture any table rows with 5 pipe-separated columns
        table_pattern = r"\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|"
        for biomarker, value, status, insight, ref in re.findall(table_pattern, table_block):
            # normalize
            biomarker_s = biomarker.strip()
            value_s = value.strip()
            status_s = status.strip()
            insight_s = insight.strip()
            ref_s = ref.strip()

            # ---------- ONLY SKIP rows where ALL five fields are empty ----------
            if not any([biomarker_s, value_s, status_s, insight_s, ref_s]):
                # This is the empty-row you showed: skip it and continue
                continue

            # ---------- ALSO SKIP rows that are pure separator artifacts ----------
            # e.g., ":-----------" or "--------" in biomarker column (common AI artifacts)
            def is_separator_cell(s: str) -> bool:
                # treat as separator if contains no alphanumeric chars
                return not bool(re.search(r"[A-Za-z0-9]", s))

            if all(is_separator_cell(c) for c in [biomarker_s, value_s, status_s, insight_s, ref_s]):
                continue

            # ---------- Append the cleaned/valid row ----------
            data["biomarker_table"].append({
                "biomarker": biomarker_s,
                "value": value_s,
                "status": status_s,
                "insight": insight_s,
                "reference_range": ref_s,
            })

    return data


# ---------------- Endpoint ----------------
@app.post("/predict")
def predict(data: BiomarkerRequest):
    """Accepts biomarker input and returns structured medical insights."""
    try:
        # --- Prompt Template ---
        prompt = """
You are an advanced **Medical Insight Generation AI** trained to analyze **biomarkers and lab results**.

⚠️ IMPORTANT — OUTPUT FORMAT INSTRUCTIONS:
Return your report in this strict markdown structure.

------------------------------
### Executive Summary
**Top 3 Health Priorities:**
1. ...
2. ...
3. ...

**Key Strengths:**
- ...
- ...

------------------------------
### System-Specific Analysis
**Cardiovascular System**
Status: Normal. Explanation: ...

**Liver Function**
Status: Elevated ALP. Explanation: ...

------------------------------
### Personalized Action Plan
**Nutrition:** ...
**Lifestyle:** ...
**Testing:** ...
**Medical Consultation:** ...

------------------------------
### Interaction Alerts
- ...
- ...

------------------------------
### Normal Ranges
- Albumin: 3.5–5.0 g/dL
- Creatinine: 0.7–1.3 mg/dL
- Glucose: 70–100 mg/dL
- CRP: 0–10 mg/L
- MCV: 80–100 fL
- RDW: 11.5–14.5 %
- ALP: 44–147 U/L
- WBC: 4.0–10.0 ×10^3/μL
- Lymphocytes: 20–40 %
- Hemoglobin: 13–17 g/dL
- PV: 2500–3000 mL

------------------------------
### Tabular Mapping
| Biomarker | Value | Status | Insight | Reference Range |
| Albumin | X | Normal | ... | 3.5–5.0 g/dL |
| Creatinine | X | High | ... | 0.7–1.3 mg/dL |
| Glucose | X | ... | ... | 70–100 mg/dL |
------------------------------
"""

        # --- Format User Data ---
        user_message = f"""
Patient Info:
- Age: {data.age}
- Gender: {data.gender}
- Height: {data.height} cm
- Weight: {data.weight} kg

Biomarkers:
- Albumin: {data.albumin} g/dL
- Creatinine: {data.creatinine} mg/dL
- Glucose: {data.glucose} mg/dL
- CRP: {data.crp} mg/L
- MCV: {data.mcv} fL
- RDW: {data.rdw} %
- ALP: {data.alp} U/L
- WBC: {data.wbc} x10^3/μL
- Lymphocytes: {data.lymphocytes} %
- Hemoglobin: {data.hb} g/dL
- Plasma Volume (PV): {data.pv} mL
"""

        # --- Gemini Call ---
        model = genai.GenerativeModel(MODEL_ID)
        response = model.generate_content(f"{prompt}\n\n{user_message}")

        if not response or not getattr(response, "text", None):
            raise ValueError("Empty response from Gemini model.")

        report_text = response.text.strip()

        # --- Parse + Clean ---
        parsed_output = parse_medical_report(report_text)
        cleaned_output = clean_json(parsed_output)

        return cleaned_output

    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
