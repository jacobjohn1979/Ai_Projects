import os, json, logging, urllib.request, urllib.error
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
log = logging.getLogger("fraud_detect.llm")

CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL   = "claude-sonnet-4-20250514"
LLM_ENABLED    = os.getenv("LLM_ANALYSIS_ENABLED","false").lower() == "true"
MAX_TEXT_CHARS = 8000

def _call_claude(system_prompt, user_message, max_tokens=1000):
    if not LLM_ENABLED:
        return {"skipped": True, "reason": "LLM_ANALYSIS_ENABLED=false"}
    try:
        body = json.dumps({
            "model": CLAUDE_MODEL, "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [{"role":"user","content":user_message}],
        }).encode("utf-8")
        req = urllib.request.Request(CLAUDE_API_URL, data=body, method="POST",
              headers={"Content-Type":"application/json","anthropic-version":"2023-06-01"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data    = json.loads(resp.read())
            content = data.get("content",[{}])[0].get("text","")
            clean   = content.strip()
            if clean.startswith("```"): clean = "\n".join(clean.split("\n")[1:])
            if clean.endswith("```"):   clean = "\n".join(clean.split("\n")[:-1])
            return json.loads(clean.strip())
    except json.JSONDecodeError as e:
        return {"error": "invalid_json_response"}
    except urllib.error.HTTPError as e:
        return {"error": f"api_http_{e.code}"}
    except Exception as e:
        return {"error": str(e)}

SINGLE_DOC_SYSTEM = """You are a senior bank fraud analyst reviewing OCR-extracted text from financial documents.
Respond with valid JSON only:
{
  "doc_type_detected": "bank_statement|payslip|tax_document|utility_bill|id_card|unknown",
  "key_fields": {"name":null,"account_number":null,"period":null,"total_income":null,"employer":null,"address":null},
  "inconsistencies": [],
  "fraud_signals": [],
  "authenticity_assessment": "genuine|suspicious|likely_fraudulent",
  "confidence": "high|medium|low",
  "reasoning": "2-3 sentence explanation",
  "risk_score_suggestion": 0
}"""

def analyze_single_document(ocr_text, doc_type_hint=""):
    flags, info = [], {}
    if not ocr_text or len(ocr_text.strip()) < 50:
        return {"llm_skipped":"insufficient_text"}, []
    hint   = f"Document type hint: {doc_type_hint}\n\n" if doc_type_hint else ""
    result = _call_claude(SINGLE_DOC_SYSTEM, f"{hint}OCR Text:\n{ocr_text[:MAX_TEXT_CHARS]}", 800)
    if result.get("skipped") or result.get("error"):
        return {"llm_status":result}, []
    info["llm_doc_type"]        = result.get("doc_type_detected","unknown")
    info["llm_key_fields"]      = result.get("key_fields",{})
    info["llm_inconsistencies"] = result.get("inconsistencies",[])
    info["llm_fraud_signals"]   = result.get("fraud_signals",[])
    info["llm_assessment"]      = result.get("authenticity_assessment","unknown")
    info["llm_confidence"]      = result.get("confidence","low")
    info["llm_reasoning"]       = result.get("reasoning","")
    info["llm_risk_score"]      = result.get("risk_score_suggestion",0)
    assessment = result.get("authenticity_assessment","")
    confidence = result.get("confidence","low")
    if assessment == "likely_fraudulent":
        flags.append("llm_assessed_likely_fraudulent")
        if confidence == "high": flags.append("llm_high_confidence_fraud")
    elif assessment == "suspicious":
        flags.append("llm_assessed_suspicious")
    if len(result.get("inconsistencies",[])) >= 3:
        flags.append("llm_multiple_inconsistencies")
    elif result.get("inconsistencies"):
        flags.append("llm_inconsistencies_detected")
    for signal in result.get("fraud_signals",[]):
        flags.append("llm_signal:" + signal[:60].lower().replace(" ","_").replace(",",""))
    return info, flags

CROSS_DOC_SYSTEM = """You are a senior bank fraud analyst. Cross-reference multiple financial documents and find inconsistencies.
Respond with valid JSON only:
{
  "cross_document_issues": [],
  "overall_consistency": "consistent|minor_issues|major_inconsistencies",
  "applicant_profile": {"likely_monthly_income":null,"income_confidence":"low","address":null,"employer":null},
  "fraud_assessment": "genuine|suspicious|likely_fraudulent",
  "reasoning": "3-4 sentence explanation",
  "recommended_action": "PASS|REVIEW|REJECT"
}"""

def analyze_cross_documents(documents):
    flags, info = [], {}
    if len(documents) < 2:
        return {"llm_cross_doc":"need_2_or_more_documents"}, []
    parts = []
    for doc_type, text in documents.items():
        if text and text.strip():
            parts.append(f"=== {doc_type.upper().replace('_',' ')} ===\n{text[:int(MAX_TEXT_CHARS/len(documents))]}")
    if not parts: return {"llm_cross_doc":"no_text"}, []
    result = _call_claude(CROSS_DOC_SYSTEM, "Applicant documents:\n\n" + "\n\n".join(parts), 1000)
    if result.get("skipped") or result.get("error"):
        return {"llm_cross_doc_status":result}, []
    issues      = result.get("cross_document_issues",[])
    consistency = result.get("overall_consistency","consistent")
    assessment  = result.get("fraud_assessment","genuine")
    recommended = result.get("recommended_action","PASS")
    info["llm_cross_doc_issues"]      = issues
    info["llm_cross_doc_consistency"] = consistency
    info["llm_applicant_profile"]     = result.get("applicant_profile",{})
    info["llm_cross_doc_assessment"]  = assessment
    info["llm_cross_doc_reasoning"]   = result.get("reasoning","")
    info["llm_recommended_action"]    = recommended
    if assessment == "likely_fraudulent": flags.append("llm_cross_doc_likely_fraudulent")
    if consistency == "major_inconsistencies": flags.append("llm_cross_doc_major_inconsistencies")
    if any(i.get("severity")=="high" for i in issues): flags.append("llm_cross_doc_high_severity_issues")
    if any("salary" in i.get("issue_type","") for i in issues): flags.append("llm_salary_mismatch_detected")
    if recommended == "REJECT": flags.append("llm_recommends_reject")
    elif recommended == "REVIEW": flags.append("llm_recommends_review")
    return info, flags

NARRATIVE_SYSTEM = """You are a compliance officer writing a brief fraud report.
Respond with valid JSON only:
{
  "executive_summary": "1-2 sentence summary",
  "risk_narrative": "3-5 sentence explanation",
  "key_concerns": [],
  "recommended_actions": [],
  "priority": "urgent|high|medium|low"
}"""

def generate_fraud_narrative(flags, doc_types, risk_score):
    if not LLM_ENABLED: return {"skipped":True}
    if not flags:
        return {"executive_summary":"No significant fraud indicators detected.",
                "risk_narrative":"Automated screening found no material fraud signals.",
                "key_concerns":[],"recommended_actions":["Standard processing may proceed."],"priority":"low"}
    prompt = f"Risk score: {risk_score}/200\nDocument types: {', '.join(doc_types)}\nFlags:\n" + "\n".join(f"- {f}" for f in flags[:30])
    result = _call_claude(NARRATIVE_SYSTEM, prompt, 600)
    return result if not result.get("error") else result

LLM_FLAG_WEIGHTS = {
    "llm_assessed_likely_fraudulent":40,"llm_high_confidence_fraud":20,
    "llm_assessed_suspicious":20,"llm_multiple_inconsistencies":15,
    "llm_inconsistencies_detected":8,"llm_cross_doc_likely_fraudulent":40,
    "llm_cross_doc_major_inconsistencies":25,"llm_salary_mismatch_detected":30,
    "llm_recommends_reject":35,"llm_recommends_review":15,
}