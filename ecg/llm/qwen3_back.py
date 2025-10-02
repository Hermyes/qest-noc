import json, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507" 

_tok = None
_mdl = None

def _load():
    global _tok, _mdl
    if _tok is None:
        _tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
        _mdl = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype="auto",
            device_map="auto",
        )
    return _tok, _mdl

def ask_label(system_prompt: str, user_prompt: str, max_new_tokens: int = 12) -> int:
    """
    Возвращает только метку 0/1. Детализированная уверенность нам не нужна.
    """
    tok, mdl = _load()
    msgs = [{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt").to(mdl.device)

    with torch.no_grad():
        out = mdl.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tok.eos_token_id,
        )

    gen = out[0, inputs.input_ids.shape[1]:]
    resp = tok.decode(gen, skip_special_tokens=True)

    m = re.search(r"\{.*\}", resp, flags=re.S)
    raw = m.group(0) if m else '{"label":0}'
    try:
        return int(json.loads(raw).get("label", 0))
    except Exception:
        return 0

def ask_label_with_raw(system_prompt: str, user_prompt: str, max_new_tokens: int = 32):
    """
    Возвращает (label:int, raw_text:str). Удобно для логов.
    """
    tok, mdl = _load()
    msgs = [{"role":"system","content":system_prompt}, {"role":"user","content":user_prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok([text], return_tensors="pt").to(mdl.device)

    with torch.no_grad():
        out = mdl.generate(
            **inp, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=0.0,
            pad_token_id=tok.eos_token_id
        )

    gen = out[0, inp.input_ids.shape[1]:]
    resp = tok.decode(gen, skip_special_tokens=True)

    import json, re
    m = re.search(r"\{.*\}", resp, flags=re.S)
    raw_json = m.group(0) if m else '{"label":0}'
    try:
        label = int(json.loads(raw_json).get("label", 0))
    except Exception:
        label = 0
    return label, resp