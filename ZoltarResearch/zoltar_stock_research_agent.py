# -*- coding: utf-8 -*-
"""
Zoltar Stock Research Agent — v4.0 (single-file drop-in build)  (2026-09-16)

Rebuild of v3.7_F after Google retired gemini-2.0-flash-exp and the Live-API transport it ran on.

What is the same as v3.7_F (the version that "felt right"):
  * six visible agent stages that build the report in front of the user, with streaming text,
    per-agent toasts, the accuracy check on Agent 1, resume-at-failed-stage retries, balloons;
  * live news / sentiment search restricted to the user's source checkboxes;
  * Zoltar SQLite grounding database (same tables, same loader), the floating bubbles,
    the background video, the model-config segmented buttons, the docx + Gmail share popover.

What changed:
  * provider layer (zr_llm.py): Google Gemini (generate_content_stream + Google Search grounding)
    or OpenAI (Responses API + web_search with allowed_domains) — pick in the sidebar or secrets;
  * each agent gets ONLY the tool it needs (DB function / web search / none) — no mixed tool sets;
  * plots: the model writes a script, the app runs it locally against the DB (server-side code
    execution never could see the SQLite file);
  * SHAP table is computed deterministically in Python; the model only writes the commentary;
  * sentiment section states its evidence ("N live sources" / "no citable sources") instead of
    silently passing model recall off as live search.

Secrets (Streamlit Cloud → App settings → Secrets, or .streamlit/secrets.toml):
    [google_api]
    api_key = "..."
    [openai]
    api_key = "..."
    [GMAIL]
    GMAIL_ACCT = "..."
    GMAIL_PASS = "..."
    [zoltar]                      # optional
    provider = "gemini"           # gemini | openai
    model = "gemini-3.8-flash"

Run locally:
    streamlit run zoltar_stock_research_agent.py
    ZOLTAR_MOCK=1 streamlit run zoltar_stock_research_agent.py   # no keys, exercises the UI
"""
import base64
import json
import os
import re
import sqlite3
import string
import random
import time
import traceback
from datetime import datetime
from io import BytesIO
from time import sleep

import matplotlib
matplotlib.use("Agg")                      # headless — must precede pyplot import
import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                         # noqa: E402
import pandas as pd                        # noqa: E402
import requests                            # noqa: E402
import seaborn as sns                      # noqa: E402
import streamlit as st                     # noqa: E402
from PIL import Image                      # noqa: E402
import json
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ======================================================================================
# [inlined] zr_llm.py — single-file build; edit the module and re-run build_dropin.py
# ======================================================================================
# --------------------------------------------------------------------------------------
# Verified model catalogue (ai.google.dev/gemini-api/docs/models + /pricing,
# developers.openai.com/api/docs/models + /pricing — fetched 2026-09-16)
# --------------------------------------------------------------------------------------
GEMINI_MODELS = {
    "gemini-3.8-flash":      "Latest Flash — $0.75 in / $3.75 out per 1M (default)",
    "gemini-3.6-flash":      "Previous Flash — $0.75 / $3.75",
    "gemini-3.5-flash-lite": "Flash-Lite — $0.30 / $2.50 (cheapest 3.x)",
    "gemini-3.1-flash-lite": "Flash-Lite — $0.25 / $1.50",
    "gemini-2.5-flash":      "2.5 Flash — $0.30 / $1.50 (still stable)",
}
OPENAI_MODELS = {
    "gpt-5.6-luna":  "Cost-optimised — $0.20 in / $1.20 out per 1M (default)",
    "gpt-5.6-terra": "Balanced — $2.00 / $12.00",
    "gpt-5.6-sol":   "Flagship — $4.00 / $20.00",
}
DEFAULT_MODEL = {"gemini": "gemini-3.8-flash", "openai": "gpt-5.6-luna"}


@dataclass
class ToolSpec:
    """A Python function the model may call. `parameters` is a JSON schema object."""
    name: str
    description: str
    parameters: Dict[str, Any]
    fn: Callable[..., Any]


@dataclass
class WebSearchSpec:
    """Provider-native live web search. allowed_domains only applies to OpenAI (Gemini has no
    domain filter — the prompt carries the restriction there, as v3.7_F did)."""
    allowed_domains: List[str] = field(default_factory=list)


@dataclass
class Citation:
    title: str
    url: str
    domain: str = ""


@dataclass
class RunResult:
    text: str = ""
    citations: List[Citation] = field(default_factory=list)
    search_queries: List[str] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    elapsed_s: float = 0.0
    error: Optional[str] = None

    @property
    def search_evidence(self) -> str:
        """Three-state summary, never a silent default (KB rule: null input => unknown state)."""
        if self.error:
            return f"unknown — run failed: {self.error}"
        if self.citations:
            return f"{len(self.citations)} live sources"
        if self.search_queries:
            return "searched, but no citable sources were returned"
        return "no live search performed"


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, default=str)
    except Exception:
        return str(obj)


def _run_tool(spec: ToolSpec, args: Dict[str, Any]) -> str:
    """Execute a tool and always return a JSON string (both APIs want a string/object payload)."""
    try:
        out = spec.fn(**(args or {}))
    except Exception as e:  # tool errors go back to the model, never crash the run
        out = {"error": f"{type(e).__name__}: {e}"}
    s = _safe_json(out)
    # Guardrail from v3.7_F: enormous SQL results blew the 1 MB websocket frame. Keep the ceiling.
    if len(s) > 400_000:
        s = s[:400_000] + " ...[truncated by app: result too large — filter/aggregate in SQL]"
    return s


# ======================================================================================
# Gemini
# ======================================================================================
class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL["gemini"]):
        from google import genai  # local import so the app can boot with only one SDK installed
        self.genai = genai
        self.types = genai.types
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def _tools(self, tools: Optional[List[ToolSpec]], web: Optional[WebSearchSpec]):
        t = self.types
        out = []
        if web is not None:
            out.append(t.Tool(google_search=t.GoogleSearch()))
        if tools:
            decls = [
                t.FunctionDeclaration(
                    name=s.name, description=s.description, parameters_json_schema=s.parameters
                )
                for s in tools
            ]
            out.append(t.Tool(function_declarations=decls))
        return out or None

    def run(self, system: str, user: str, tools: Optional[List[ToolSpec]] = None,
            web_search: Optional[WebSearchSpec] = None, temperature: float = 0.1,
            top_p: float = 0.9, on_text: Optional[Callable[[str], None]] = None,
            on_event: Optional[Callable[[str, Any], None]] = None,
            max_tool_rounds: int = 12) -> RunResult:
        t = self.types
        t0 = time.time()
        res = RunResult(provider=self.name, model=self.model)
        by_name = {s.name: s for s in (tools or [])}
        config = t.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            top_p=top_p,
            tools=self._tools(tools, web_search),
            automatic_function_calling=t.AutomaticFunctionCallingConfig(disable=True),
        )
        contents: List[Any] = [t.Content(role="user", parts=[t.Part(text=user)])]

        try:
            for _round in range(max_tool_rounds + 1):
                model_parts: List[Any] = []
                fcalls: List[Any] = []
                stream = self.client.models.generate_content_stream(
                    model=self.model, contents=contents, config=config
                )
                for chunk in stream:
                    cand = (chunk.candidates or [None])[0]
                    if cand is None:
                        continue
                    if cand.content and cand.content.parts:
                        for p in cand.content.parts:
                            model_parts.append(p)
                            if p.function_call:
                                fcalls.append(p.function_call)
                            elif p.text and not getattr(p, "thought", False):
                                res.text += p.text
                                if on_text:
                                    on_text(p.text)
                    gm = cand.grounding_metadata
                    if gm:
                        for q in (gm.web_search_queries or []):
                            if q not in res.search_queries:
                                res.search_queries.append(q)
                                if on_event:
                                    on_event("search", q)
                        for gc in (gm.grounding_chunks or []):
                            w = gc.web
                            if w and w.uri and all(c.url != w.uri for c in res.citations):
                                res.citations.append(Citation(w.title or w.uri, w.uri, w.domain or ""))
                if not fcalls:
                    break
                # echo the model turn verbatim (keeps thought_signature), then answer every call
                contents.append(t.Content(role="model", parts=model_parts))
                resp_parts = []
                for fc in fcalls:
                    args = dict(fc.args or {})
                    if on_event:
                        on_event("tool_call", {"name": fc.name, "args": args})
                    spec = by_name.get(fc.name)
                    out = _run_tool(spec, args) if spec else _safe_json({"error": f"unknown tool {fc.name}"})
                    res.tool_calls.append({"name": fc.name, "args": args, "result_chars": len(out)})
                    if on_event:
                        on_event("tool_result", {"name": fc.name, "preview": out[:300]})
                    resp_parts.append(t.Part.from_function_response(name=fc.name, response={"result": out}))
                contents.append(t.Content(role="user", parts=resp_parts))
            else:
                res.error = f"stopped after {max_tool_rounds} tool rounds"
        except Exception as e:
            res.error = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        res.elapsed_s = time.time() - t0
        return res


# ======================================================================================
# OpenAI (Responses API over plain HTTPS — no `openai` package needed)
# ======================================================================================
# Why REST instead of the SDK: the ZoltarFinancial root requirements.txt pins openai==0.28 (shared by
# several apps). 0.28 has no Responses API and a newer pin would break the other apps, so this
# provider speaks the wire protocol directly with `requests`, which is already a dependency.
# Wire facts (checked against openai-python 3.14 _streaming.py / types on 2026-09-16): SSE frames are
# "event: <type>\ndata: <json>\n\n"; every data JSON carries "type"; text arrives as
# response.output_text.delta {delta}; tool calls as response.output_item.done {item:{type:'function_call',
# call_id,name,arguments}}; web searches as item.type 'web_search_call' {action:{query}}; the final
# object as response.completed {response:{id,output:[{type:'message',content:[{annotations:[{type:
# 'url_citation',url,title}]}]}]}}.
OPENAI_URL = "https://api.openai.com/v1/responses"


def _iter_sse(resp):
    """Yield parsed JSON objects from an SSE response (requests, stream=True)."""
    data_lines: List[str] = []
    for raw in resp.iter_lines(decode_unicode=True):
        if raw is None:
            continue
        line = raw.strip("\r")
        if line == "":
            if data_lines:
                payload = "\n".join(data_lines)
                data_lines = []
                if payload.startswith("[DONE]"):
                    return
                try:
                    yield json.loads(payload)
                except Exception:
                    pass
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
        # "event:" and comment lines are ignored — the JSON's own "type" is authoritative
    if data_lines:
        try:
            yield json.loads("\n".join(data_lines))
        except Exception:
            pass


class OpenAIProvider:
    name = "openai"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL["openai"], timeout: float = 300.0):
        import requests
        self.requests = requests
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def _tools(self, tools: Optional[List[ToolSpec]], web: Optional[WebSearchSpec]):
        out: List[Dict[str, Any]] = []
        if web is not None:
            ws: Dict[str, Any] = {"type": "web_search"}
            if web.allowed_domains:
                ws["filters"] = {"allowed_domains": web.allowed_domains[:100]}
            out.append(ws)
        for s in (tools or []):
            out.append({"type": "function", "name": s.name, "description": s.description,
                        "parameters": s.parameters})
        return out or None

    def _post(self, body: Dict[str, Any]):
        r = self.requests.post(OPENAI_URL, json=body, stream=True, timeout=self.timeout,
                               headers={"Authorization": f"Bearer {self.api_key}",
                                        "Content-Type": "application/json", "Accept": "text/event-stream"})
        if r.status_code >= 400:
            try:
                msg = r.json().get("error", {}).get("message", r.text[:300])
            except Exception:
                msg = r.text[:300]
            raise RuntimeError(f"OpenAI HTTP {r.status_code}: {msg}")
        return r

    def run(self, system: str, user: str, tools: Optional[List[ToolSpec]] = None,
            web_search: Optional[WebSearchSpec] = None, temperature: float = 0.1,
            top_p: float = 0.9, on_text: Optional[Callable[[str], None]] = None,
            on_event: Optional[Callable[[str, Any], None]] = None,
            max_tool_rounds: int = 12) -> RunResult:
        t0 = time.time()
        res = RunResult(provider=self.name, model=self.model)
        by_name = {s.name: s for s in (tools or [])}
        tool_defs = self._tools(tools, web_search)
        input_items: List[Any] = [{"role": "user", "content": user}]
        prev_id: Optional[str] = None
        sampling: Dict[str, Any] = {"temperature": temperature, "top_p": top_p}

        try:
            for _round in range(max_tool_rounds + 1):
                body: Dict[str, Any] = dict(model=self.model, instructions=system, input=input_items,
                                            stream=True, store=True, **sampling)
                if tool_defs:
                    body["tools"] = tool_defs
                if prev_id:
                    body["previous_response_id"] = prev_id
                try:
                    r = self._post(body)
                except RuntimeError as e:
                    # gpt-5.x reasoning tiers may reject sampling params — retry once without them
                    if sampling and ("temperature" in str(e) or "top_p" in str(e)):
                        sampling = {}
                        body.pop("temperature", None); body.pop("top_p", None)
                        r = self._post(body)
                    else:
                        raise

                fcalls: List[Dict[str, Any]] = []
                final: Optional[Dict[str, Any]] = None
                with r:
                    for ev in _iter_sse(r):
                        et = ev.get("type", "")
                        if et == "response.output_text.delta":
                            d = ev.get("delta", "")
                            res.text += d
                            if on_text and d:
                                on_text(d)
                        elif et == "response.output_item.done":
                            item = ev.get("item") or {}
                            it = item.get("type", "")
                            if it == "function_call":
                                fcalls.append({"call_id": item.get("call_id"), "name": item.get("name"),
                                               "arguments": item.get("arguments")})
                            elif it == "web_search_call":
                                q = (item.get("action") or {}).get("query")
                                if q and q not in res.search_queries:
                                    res.search_queries.append(q)
                                    if on_event:
                                        on_event("search", q)
                        elif et == "response.completed":
                            final = ev.get("response") or {}
                        elif et in ("error", "response.failed"):
                            err = ev.get("error") or (ev.get("response") or {}).get("error") or ev
                            raise RuntimeError(f"OpenAI stream error: {err}")
                if final:
                    prev_id = final.get("id")
                    for item in final.get("output") or []:
                        if item.get("type") == "message":
                            for c in item.get("content") or []:
                                for a in c.get("annotations") or []:
                                    if a.get("type") == "url_citation":
                                        url = a.get("url", "")
                                        if url and all(x.url != url for x in res.citations):
                                            res.citations.append(Citation(a.get("title") or url, url))
                if not fcalls:
                    break
                input_items = []
                for fc in fcalls:
                    try:
                        args = json.loads(fc["arguments"] or "{}")
                    except Exception:
                        args = {}
                    if on_event:
                        on_event("tool_call", {"name": fc["name"], "args": args})
                    spec = by_name.get(fc["name"])
                    out = _run_tool(spec, args) if spec else _safe_json({"error": f"unknown tool {fc['name']}"})
                    res.tool_calls.append({"name": fc["name"], "args": args, "result_chars": len(out)})
                    if on_event:
                        on_event("tool_result", {"name": fc["name"], "preview": out[:300]})
                    input_items.append({"type": "function_call_output", "call_id": fc["call_id"], "output": out})
            else:
                res.error = f"stopped after {max_tool_rounds} tool rounds"
        except Exception as e:
            res.error = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        res.elapsed_s = time.time() - t0
        return res


# ======================================================================================
# Mock (offline smoke tests / UI development without keys)
# ======================================================================================
class MockProvider:
    name = "mock"

    def __init__(self, api_key: str = "", model: str = "mock"):
        self.model = model

    def run(self, system, user, tools=None, web_search=None, temperature=0.1, top_p=0.9,
            on_text=None, on_event=None, max_tool_rounds=12) -> RunResult:
        res = RunResult(provider=self.name, model=self.model)
        # exercise one tool call if a tool is offered, so the app's tool plumbing is covered
        if tools:
            spec = tools[0]
            args = {"sql": "SELECT name FROM sqlite_master WHERE type='table' LIMIT 3"} \
                if spec.name == "execute_query" else {}
            if on_event:
                on_event("tool_call", {"name": spec.name, "args": args})
            out = _run_tool(spec, args)
            res.tool_calls.append({"name": spec.name, "args": args, "result_chars": len(out)})
            if on_event:
                on_event("tool_result", {"name": spec.name, "preview": out[:300]})
        if web_search is not None:
            res.search_queries.append("mock query")
            res.citations.append(Citation("Mock source", "https://example.com/mock", "example.com"))
        body = "[MOCK " + self.model + "] " + user[:400].replace("\n", " ")
        if "OUTPUT_PATH" in (system or ""):
            # a plotting request: return runnable code so the app's local-exec path is exercised
            body = ("```python\n"
                    "df = query(\"SELECT Symbol, Date, Close_Price FROM high_risk WHERE Symbol IN ('AAPL','MSFT') ORDER BY Date\")\n"
                    "fig, ax = plt.subplots(1, 1, figsize=(8, 5))\n"
                    "for s, g in df.groupby('Symbol'):\n"
                    "    ax.plot(pd.to_datetime(g['Date']), g['Close_Price'], label=s)\n"
                    "ax.legend(); ax.tick_params(axis='x', rotation=-45)\n"
                    "plt.tight_layout(); plt.savefig(OUTPUT_PATH, dpi=110)\n"
                    "```\nReferences to visualization\nMock chart of close prices.")
        elif "SYMBOLS:" in user:
            body += "\n\nSYMBOLS: AAPL, MSFT, NVDA"
        words = (body + " ... May the riches be with you...").split(" ")
        for w in words:
            piece = w + " "
            res.text += piece
            if on_text:
                on_text(piece)
            time.sleep(0.005)
        return res


def make_provider(kind: str, api_key: str, model: Optional[str] = None):
    kind = (kind or "").lower()
    if kind == "gemini":
        return GeminiProvider(api_key, model or DEFAULT_MODEL["gemini"])
    if kind == "openai":
        return OpenAIProvider(api_key, model or DEFAULT_MODEL["openai"])
    if kind == "mock":
        return MockProvider(api_key, model or "mock")
    raise ValueError(f"unknown provider '{kind}' (expected gemini | openai | mock)")


# ======================================================================================
# [inlined] zr_legacy_ui.py — single-file build; edit the module and re-run build_dropin.py
# ======================================================================================
def set_bg_video(video_file):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: transparent !important;
        }}
        .block-container {{
            background: transparent !important;
        }}
        .main {{
            background: transparent !important;
        }}
        video.bgvid {{
            position: fixed;
            top: 50%;
            left: 50%;
            min-width: 100vw;
            min-height: 100vh;
            width: auto;
            height: auto;
            z-index: -1;
            object-fit: cover;
            opacity: 0.7;
            pointer-events: none;
            /* Zoom in by scaling the video */
            transform: translate(-50%, -50%) scale(1.27);  /* change 1.2 to any zoom factor you want */
        }}
        </style>
        <video autoplay loop muted class="bgvid">
            <source src="data:video/mp4;base64,{video_file}" type="video/mp4">
        </video>
        """,
        unsafe_allow_html=True
    )


def generate_top_10_stream(db_path='zoltar_financial.db', top_n1=5, top_n2=5):
    conn = sqlite3.connect(db_path)
    
    # try:
    #     # Get latest date
    #     latest_date = conn.execute(
    #         "SELECT MAX(Date) FROM low_risk"
    #     ).fetchone()[0]
        
    #     # Get top 10 symbols from low_risk
    #     top_symbols = conn.execute(f"""
    #         SELECT Symbol, Score as Low_Risk_Score 
    #         FROM low_risk 
    #         WHERE Date = '{latest_date}'
    #         ORDER BY Low_Risk_Score DESC 
    #         LIMIT 10
    #     """).fetchall()
    try:
        # Get latest date
        latest_date = conn.execute(
            "SELECT MAX(Date) FROM low_risk"
        ).fetchone()[0]
        
        # Get top 10 symbols from low_risk
        top_low = conn.execute(f"""
            SELECT Symbol, Score as Low_Risk_Score 
            FROM low_risk 
            WHERE Date = '{latest_date}'
            GROUP BY 1,2
            ORDER BY Low_Risk_Score DESC 
            LIMIT {top_n1}
        """).fetchall()
        
        # Get top 10 symbols from low_risk
        top_high = conn.execute(f"""
             SELECT Symbol, Score as High_Risk_Score 
             FROM high_risk 
             WHERE Date = '{latest_date}'
             GROUP BY 1,2
             ORDER BY High_Risk_Score DESC 
             LIMIT {top_n2}
        """).fetchall()
        top_symbols = top_low + top_high

        # Combine, avoiding duplicates (keep order: low_risk first, then high_risk additions)
        symbols_seen = set()
        combined = []
        for symbol, score in top_low + top_high:
            if symbol not in symbols_seen:
                symbols_seen.add(symbol)
                combined.append(symbol)        
        stream_content = []
        
        for symbol, _ in top_symbols:
            try:
                # Always get the low risk score for this symbol
                low_data = conn.execute(f"""
                    SELECT Score FROM low_risk 
                    WHERE Symbol = '{symbol}' AND Date = '{latest_date}'
                """).fetchone()
                low_score = low_data[0] if low_data else None
        
                # Get high risk data
                high_data = conn.execute(f"""
                    SELECT Score as High_Risk_Score, Score_HoldPeriod as High_Risk_Score_HoldPeriod 
                    FROM high_risk 
                    WHERE Symbol = '{symbol}' AND Date = '{latest_date}'
                """).fetchone()
                
                # Get fundamentals
                fundamentals = conn.execute(f"""
                    SELECT Fundamentals_Industry, Fundamentals_Sector,
                           Fundamentals_PE, Fundamentals_PB,
                           Fundamentals_Dividends, Fundamentals_ExDividendDate,
                           Fundamentals_MarketCap, Fundamentals_Description
                    FROM fundamentals 
                    WHERE Symbol = '{symbol}'
                """).fetchone()
                
                if not high_data or not fundamentals:
                    continue
                
                # Unpack data
                high_score, hold_period = high_data
                (industry, sector, pe, pb, 
                 dividend, ex_div, mcap, desc) = fundamentals
                
                # Format values
                dividend_pct = f"{dividend:.2f}%" if dividend else "none"
                ex_div_date = pd.to_datetime(ex_div).strftime('%m-%d-%Y') if ex_div else 'N/A'
                mcap_formatted = f"${mcap/1e9:.2f}B" if mcap else 'N/A'
                truncated_desc = f"{desc[:300]}..." if desc else ""
                
                stream_content.append({
                    "symbol": symbol,
                    "low_score": f"{low_score:.2%}" if low_score is not None else "N/A",
                    "high_score": f"{high_score:.2%}",
                    "hold_period": f"{hold_period:.0f}d",
                    "industry": industry,
                    "sector": sector,
                    "pe": f"{pe:.2f}",
                    "pb": f"{pb:.2f}",
                    "dividend": dividend_pct,
                    "ex_div": ex_div_date,
                    "mcap": mcap_formatted,
                    "desc": truncated_desc
                })
                
            except Exception as e:
                print(f"Error processing {symbol}: {str(e)}")
                
        return stream_content
        
    finally:
        conn.close()


def display_bubbles(col, items):
    html = "<div class='bubble-container'>"
    n = len(items)
    base_height = 750  # px
    container_height = base_height + int((n-5)*150)
    bubble_diameter = 150    # px  <-- updated

    if n > 1:
        space_between = (container_height - bubble_diameter) // (n - 1)
    else:
        space_between = 0

    for i, item in enumerate(items):
        container_width = 200  # px, set to your actual container width
        max_left_percent = 100 - (bubble_diameter / container_width * 100)
        left = random.uniform(0, max_left_percent)
        hue = random.randint(0, 360)
        gradient = (
            f"radial-gradient(circle at 35% 30%, "
            f"hsla({hue}, 80%, 22%, 0.95) 0%, "
            f"hsla({hue}, 80%, 14%, 0.92) 55%, "
            f"hsla({hue}, 90%, 7%, 0.95) 100%)"
        )
        top_px = i * space_between + random.randint(-8, 8)
        duration = random.uniform(2.5, 5.5)
        delay = random.uniform(0, 2)
        html += f"""
            <div class="bubble" style="
                left: {left}%;
                top: {top_px}px;
                width: {bubble_diameter}px;
                height: {bubble_diameter}px;
                background: {gradient};
                animation-duration: {duration}s;
                animation-delay: {delay}s;
            ">
                <h3 style='color: hsl({hue}, 60%, 70%); font-size:0.9em; margin:0; padding:0;'>{item['symbol']}</h3>
                <div class="bubble-desc-scroll-x" style="margin:0; padding:0; margin-top:2px;">
                    <div class="bubble-desc-scroll-x-inner" style="font-size:0.8em;">
                        {item['desc']} &nbsp;&nbsp;&nbsp; {item['desc']}
                    </div>
                </div>
                <p style='font-size: 0.7em; margin:0; padding:0 6px; text-align:center; word-break:break-word; color: #e0e0e0;'>
                    🏭 {item['industry']}<br>
                    🚀 High Rank: {item['high_score']}<br>
                    📈 Low Rank: {item['low_score']}<br>
                    💰 P/E: {item['pe']} | P/B: {item['pb']}<br>
                    📅 Ex-Div: {item['ex_div']}<br>
                    💵 Div: {item['dividend']}
                </p>
            </div>
        """
        html += "</div>"
    html += "</div>"
    col.markdown(bubble_style() + html, unsafe_allow_html=True)


# def bubble_style():
#     return """
#     <style>
#         @keyframes float {
#             0%   { transform: translateY(0px);}
#             50%  { transform: translateY(-20px);}
#             100% { transform: translateY(0px);}
#         }
#         @keyframes focusBlur {
#             0%   { filter: blur(2.5px);}
#             25%  { filter: blur(2.5px);}
#             35%  { filter: blur(0.5px);}
#             65%  { filter: blur(0.5px);}
#             75%  { filter: blur(2.5px);}
#             100% { filter: blur(2.5px);}
#         }
#         @keyframes scroll-horizontal {
#             0%   { transform: translateX(0%);}
#             100% { transform: translateX(-50%);}
#         }
#         .bubble-container {
#             position: relative;
#             width: 100%;
#             max-width: 100%;
#             height: 1100px;
#             box-sizing: border-box;
#         }
#         .bubble {
#             border-radius: 50%;
#             margin: 10px;
#             position: absolute;
#             animation: float 6s ease-in-out infinite, focusBlur 6s ease-in-out infinite;
#             backdrop-filter: blur(5px);
#             border: 1px solid rgba(255,255,255,0.18);
#             box-shadow: 0 8px 24px rgba(0,0,0,0.14), 0 1.5px 8px 2px rgba(255,255,255,0.08) inset;
#             transition: transform 0.3s ease;
#             display: flex;
#             flex-direction: column;
#             align-items: center;
#             justify-content: center;
#             overflow: hidden;
#         }
#         .bubble-desc-scroll-x {
#             width: 90%;
#             height: 2em;
#             overflow: hidden;
#             background: transparent;
#             margin: 0 auto;
#             position: relative;
#             white-space: nowrap;
#         }
#         .bubble-desc-scroll-x-inner {
#             display: inline-block;
#             white-space: nowrap;
#             animation: scroll-horizontal 18s linear infinite;
#         }
#     </style>
#     """


def bubble_style():
    return """
    <style>
        @keyframes float {
            0%   { transform: translateY(0px);}
            50%  { transform: translateY(-20px);}
            100% { transform: translateY(0px);}
        }
        @keyframes focusBlur {
            0%   { filter: blur(2.5px);}
            25%  { filter: blur(2.5px);}
            35%  { filter: blur(0.5px);}
            65%  { filter: blur(0.5px);}
            75%  { filter: blur(2.5px);}
            100% { filter: blur(2.5px);}
        }
        @keyframes scroll-horizontal {
            0%   { transform: translateX(0%);}
            100% { transform: translateX(-50%);}
        }
        .bubble-container {
            position: relative;
            width: 100%;
            max-width: 100%;
            height: 1100px;
            box-sizing: border-box;
        }
        .bubble {
            border-radius: 50%;
            margin: 10px;
            position: absolute;
            animation: float 6s ease-in-out infinite;
            backdrop-filter: blur(5px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.14), 0 1.5px 8px 2px rgba(255,255,255,0.08) inset;
            transition: transform 0.3s ease;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }
        .bubble-desc-scroll-x {
            width: 90%;
            height: 2em;
            overflow: hidden;
            background: transparent;
            margin: 0 auto;
            position: relative;
            white-space: nowrap;
        }
        .bubble-desc-scroll-x-inner {
            display: inline-block;
            white-space: nowrap;
            animation: scroll-horizontal 37s linear infinite;
        }
    </style>
    """


def segmented_buttons(label, levels, key_prefix):
    cols = st.sidebar.columns(len(levels))
    selected = st.session_state.get(f"{key_prefix}_selected", levels[0][0])
    for i, (level, _) in enumerate(levels):
        button_kwargs = {}
        if selected == level:
            button_kwargs["type"] = "primary"
        if cols[i].button(level, key=f"{key_prefix}_{level}", **button_kwargs):
            st.session_state[f"{key_prefix}_selected"] = level
            selected = level
    return dict(levels)[selected]


def is_blank_png(img_bytes):
    # Quick check for empty or very small files
    if not img_bytes or len(img_bytes) < 8500:
        return True

    try:
        with Image.open(BytesIO(img_bytes)) as img:
            img = img.convert("RGBA")  # Ensure 4 channels
            extrema = img.getextrema()  # Returns (min, max) for each channel

            # Check if all pixels are fully transparent
            if extrema[3] == (0, 0):
                return True

            # Check if all pixels are white (255,255,255,255)
            if all(channel == (255, 255) for channel in extrema):
                return True

            # For grayscale/other modes, check if all pixels are same value
            if all(e[0] == e[1] for e in extrema):
                return True

            # Optionally, check if all pixels are the same color
            if len(set(img.getdata())) == 1:
                return True

    except Exception as e:
        # If PIL fails to open, treat as blank/invalid
        print(f"Image validation error: {e}")
        return True

    return False


APP_VERSION = "4.0"
PLOT_PATH = "stock_price_plot.png"


def show_image(data, caption=""):
    """st.image width API changed in 1.50 (width='stretch'); support the repo's 1.36 pin too."""
    try:
        major, minor = (int(x) for x in st.__version__.split(".")[:2])
    except Exception:
        major, minor = 1, 36
    if (major, minor) >= (1, 50):
        st.image(data, caption=caption, width="stretch")
    elif (major, minor) >= (1, 40):
        st.image(data, caption=caption, use_container_width=True)
    else:  # 1.36 (repo pin)
        st.image(data, caption=caption, use_column_width=True)

# ======================================================================================
# Page + look
# ======================================================================================
try:
    favicon = "https://github.com/apod-1/ZoltarFinancial/raw/main/docs/ZoltarSurf_48x48.png"
except (KeyError, FileNotFoundError):
    favicon = st.secrets["browser"]["favicon"]
st.set_page_config(page_title="Zoltar Stock Research Agent", page_icon=favicon, layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown("""<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}</style>""",
            unsafe_allow_html=True)


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def _bg_video_b64() -> str:
    try:
        r = requests.get("https://github.com/apod-1/ZoltarFinancial/raw/main/docs/wave_vid.mp4", timeout=20)
        r.raise_for_status()
        return base64.b64encode(r.content).decode()
    except Exception as e:
        print(f"background video unavailable: {e}")
        return ""


_vid = _bg_video_b64()
if _vid:
    set_bg_video(_vid)

col1, col2, col3 = st.columns([1, 5, 1])
with col2:
    st.title("US Equities Zoltar Research Agent 🤖",
             help="I am here to help you make better decisions! Don't be shy - ask away...")

# ======================================================================================
# Secrets / keys
# ======================================================================================
def _secrets_file_present() -> bool:
    """Touching st.secrets with no secrets.toml prints an error box on Streamlit 1.36; skip it locally."""
    try:
        from streamlit.runtime.secrets import SECRETS_FILE_LOCS
        return any(os.path.exists(p) for p in SECRETS_FILE_LOCS)
    except Exception:
        return True


HAVE_SECRETS = _secrets_file_present()


def _secret(section: str, key: str, env: str = "") -> str:
    if HAVE_SECRETS:
        try:
            v = st.secrets[section][key]
            if v:
                return str(v)
        except Exception:
            pass
    return os.getenv(env, "") if env else ""


GOOGLE_API_KEY = _secret("google_api", "api_key", "GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = _secret("openai", "api_key", "OPENAI_API_KEY")
CFG_PROVIDER = (_secret("zoltar", "provider", "ZOLTAR_PROVIDER") or "").lower()
CFG_MODEL = _secret("zoltar", "model", "ZOLTAR_MODEL")
MOCK_MODE = os.getenv("ZOLTAR_MOCK", "") == "1"

# ======================================================================================
# Session state
# ======================================================================================
for k, v in {
    "final_agent_result": "",
    "image": None,
    "temp_selected": "0.1 - Middle",
    "top_p_selected": "0.9 - Middle",
    "agent_repo": {"agents": {}, "execution_order": []},
    "agent_progress": {},
    "last_run_meta": {},
}.items():
    st.session_state.setdefault(k, v)

# ======================================================================================
# Database (unchanged semantics from v3.7_F)
# ======================================================================================
DATA_ROOT = os.getenv("ZOLTAR_DATA_ROOT", "/mount/src/zoltarfinancial")


def random_db_filename(base_name="zoltar_financial.db"):
    name, ext = os.path.splitext(base_name)
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{name}_{suffix}{ext}"


def get_sqlite_connection_with_random_on_lock(db_file, max_retries=3, retry_delay=0.5):
    for attempt in range(max_retries):
        try:
            conn = sqlite3.connect(db_file, timeout=10, check_same_thread=False)
            conn.execute("PRAGMA quick_check;")
            return conn, db_file
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                db_file = random_db_filename(db_file)
                sleep(retry_delay)
            else:
                raise
    raise RuntimeError("Could not acquire database connection after multiple retries (database is locked).")


def get_latest_file(data_dir, prefix):
    try:
        files = [f for f in os.listdir(data_dir) if f.startswith(prefix) and f.endswith(".pkl")]
        if not files:
            return None
        return os.path.join(data_dir, max(files, key=lambda x: os.path.getmtime(os.path.join(data_dir, x))))
    except FileNotFoundError:
        return None


LOAD_PLAN = [
    ("daily_ranks", "all_high_risk_PROD", "all_high_risk"),
    ("daily_ranks", "all_low_risk_PROD", "all_low_risk"),
    ("daily_ranks", "high_risk_PROD", "high_risk"),
    ("daily_ranks", "low_risk_PROD", "low_risk"),
    ("data", "fundamentals_df", "fundamentals"),
    ("data", "ratings_detail_df", "ratings_detail"),
    ("daily_ranks", "combined_SHAP_summary_Large", "shap_summary_Large"),
    ("daily_ranks", "combined_SHAP_summary_Mid", "shap_summary_Mid"),
    ("daily_ranks", "combined_SHAP_summary_Small", "shap_summary_Small"),
]
SHAP_TABLES = ("shap_summary_Large", "shap_summary_Mid", "shap_summary_Small")


def load_data_into_db(conn) -> dict:
    """Returns {table: row_count | None}. None means 'no source file found' (unknown, not zero)."""
    status = {}
    for sub, prefix, table in LOAD_PLAN:
        path = get_latest_file(os.path.join(DATA_ROOT, sub), prefix)
        if not path:
            status[table] = None
            continue
        try:
            df = pd.read_pickle(path)
            if table == "ratings_detail":
                df["RatingText"] = df["RatingText"].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
                df["RatingPublishedAt"] = pd.to_datetime(df["RatingPublishedAt"], errors="coerce")
            if table in SHAP_TABLES:
                if "Feature Category" in df.columns:
                    df = df.drop(columns=["Feature Category"])
                if df.index.name is not None or not df.index.equals(pd.RangeIndex(len(df))):
                    df = df.reset_index()
                df = df.rename(columns={"index": "Symbol"})
            with conn:
                df.to_sql(table, conn, if_exists="replace", index=False)
            status[table] = int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
        except Exception as e:
            print(f"Error inserting into {table}: {e}")
            status[table] = None
    return status


@st.cache_resource(show_spinner="Updating Zoltar database...")
def open_db():
    conn, used = get_sqlite_connection_with_random_on_lock("zoltar_financial.db")
    status = load_data_into_db(conn)
    return conn, used, status


db_conn, db_file_used, db_status = open_db()
DB_TABLES_LOADED = [t for t, n in db_status.items() if n]
DB_TABLES_MISSING = [t for t, n in db_status.items() if not n]


def execute_query(sql: str) -> dict:
    """Run a read-only SQL SELECT against the Zoltar database and return rows."""
    if not re.match(r"^\s*(select|with|pragma)\b", sql or "", re.I):
        return {"call": f"execute_query({sql})", "error": "Only SELECT / WITH / PRAGMA statements are allowed."}
    cur = db_conn.cursor()
    cur.execute(sql)
    cols = [d[0] for d in cur.description] if cur.description else []
    rows = cur.fetchmany(2000)
    return {"call": f"execute_query({sql})", "columns": cols, "results": rows,
            "truncated": len(rows) == 2000}


def query_df(sql: str) -> pd.DataFrame:
    """DataFrame helper injected into model-written plotting scripts."""
    return pd.read_sql_query(sql, db_conn)


EXECUTE_QUERY_TOOL = ToolSpec(
    name="execute_query",
    description="Execute a read-only SQLite SELECT query against the Zoltar Ranks database and return "
                "columns and rows (max 2000 rows — filter and aggregate in SQL).",
    parameters={"type": "object", "properties": {"sql": {"type": "string", "description": "SQLite SELECT statement"}},
                "required": ["sql"]},
    fn=execute_query,
)


def known_symbols() -> set:
    try:
        return {r[0] for r in db_conn.execute("SELECT DISTINCT Symbol FROM high_risk").fetchall() if r[0]}
    except Exception:
        return set()


# ======================================================================================
# Sidebar
# ======================================================================================
with st.sidebar:
    st.sidebar.markdown(
        """
        <style>
        .zoltar-btn { background: linear-gradient(135deg, #301934 0%, #9370DB 100%); border: none; color: #fff;
            padding: 14px 28px; text-align: center; display: inline-block; font-size: 18px; font-weight: 600;
            margin: 8px 2px; border-radius: 12px; cursor: pointer;
            box-shadow: 0 4px 14px 0 rgba(80, 40, 120, 0.45), 0 1.5px 8px 2px rgba(255,255,255,0.06) inset;
            transition: all 0.18s cubic-bezier(.4,0,.2,1); }
        .zoltar-btn:hover { background: linear-gradient(135deg, #9370DB 0%, #301934 100%); transform: translateY(-2px) scale(1.04); }
        .disclaimer-btn { background: linear-gradient(135deg, #301934 0%, #9370DB 100%); border: none; color: #fff;
            padding: 10px 22px; text-align: center; display: inline-block; font-size: 15px; font-weight: 600;
            margin: 10px 2px; border-radius: 10px; cursor: pointer; box-shadow: 0 4px 14px 0 rgba(80, 40, 120, 0.25); }
        </style>
        <a href="https://zoltar.streamlit.app" target="_blank"><button class="zoltar-btn">Open Zoltar Research Platform</button></a>
        """, unsafe_allow_html=True)

    show_top_symbols = st.sidebar.toggle("Show Top Symbols Section", value=True)
    top_n1, top_n2 = 5, 5
    with st.expander("Bubble Display Settings", expanded=False):
        if show_top_symbols:
            c1, c2 = st.columns(2)
            top_n1 = c1.number_input("Symbols for Low Rank", 1, 20, 5, 1)
            top_n2 = c2.number_input("Symbols for High Rank", 1, 20, 5, 1)

    # ---- Engine ----
    st.sidebar.header("AI Engine")
    engines = []
    if GOOGLE_API_KEY:
        engines.append("gemini")
    if OPENAI_API_KEY:
        engines.append("openai")
    if MOCK_MODE or not engines:
        engines.append("mock")
    default_engine = CFG_PROVIDER if CFG_PROVIDER in engines else engines[0]
    provider_kind = st.radio("Provider", engines, index=engines.index(default_engine), horizontal=True,
                             format_func=lambda k: {"gemini": "Google Gemini", "openai": "OpenAI", "mock": "Mock (no keys)"}[k])
    if provider_kind == "gemini":
        opts = list(GEMINI_MODELS)
        model_name = st.selectbox("Model", opts, index=opts.index(CFG_MODEL) if CFG_MODEL in opts else 0,
                                  format_func=lambda m: f"{m} — {GEMINI_MODELS[m]}")
    elif provider_kind == "openai":
        opts = list(OPENAI_MODELS)
        model_name = st.selectbox("Model", opts, index=opts.index(CFG_MODEL) if CFG_MODEL in opts else 0,
                                  format_func=lambda m: f"{m} — {OPENAI_MODELS[m]}")
    else:
        model_name = "mock"
        if not (GOOGLE_API_KEY or OPENAI_API_KEY):
            st.warning("No API key found in secrets — running in mock mode.")

    # ---- Agent configuration (unchanged options) ----
    st.sidebar.header("Agent Configuration")
    st.sidebar.write("**News sources selection:**")
    c1s, c2s = st.sidebar.columns(2)
    with c1s:
        google_trends = st.checkbox("Google Trends", value=False)
        stocktwits = st.checkbox("StockTwits", value=True)
        zacks = st.checkbox("Zacks", value=False)
        seeking = st.checkbox("SeekingAlpha", value=True)
    with c2s:
        reddit = st.checkbox("Reddit", value=True)
        yahoo = st.checkbox("Yahoo Finance", value=False)
        tipranks = st.checkbox("TipRanks", value=True)
        nasdaq = st.checkbox("NASDAQ", value=True)

    SOURCES = [  # (enabled, label for the prompt, domain for OpenAI allowed_domains)
        (google_trends, "Google Trends (https://trends.google.com/)", "trends.google.com"),
        (stocktwits, "StockTwits (https://stocktwits.com/)", "stocktwits.com"),
        (yahoo, "Yahoo Finance (https://finance.yahoo.com/)", "finance.yahoo.com"),
        (tipranks, "TipRanks (https://www.tipranks.com/)", "tipranks.com"),
        (zacks, "Zacks (https://www.zacks.com/)", "zacks.com"),
        (reddit, "Reddit (https://www.reddit.com/)", "reddit.com"),
        (seeking, "SeekingAlpha (https://seekingalpha.com/)", "seekingalpha.com"),
        (nasdaq, "NASDAQ.com (https://www.nasdaq.com/market-activity/stocks)", "nasdaq.com"),
    ]
    selected_sources = [lbl for on, lbl, _ in SOURCES if on]
    selected_domains = [dom for on, _, dom in SOURCES if on]
    source_str = ", ".join(selected_sources) if selected_sources else "no sources selected"
    strict_domains = st.checkbox("Restrict search strictly to selected sites (OpenAI only)", value=False,
                                 help="Gemini's Google Search has no domain filter; the restriction is carried in the prompt.")

    st.sidebar.write("**Visualization selection:**")
    c1v, c2v = st.sidebar.columns(2)
    with c1v:
        Pie_chart = st.checkbox("Pie Chart", value=False)
        Return_hold = st.checkbox("Returns", value=True)
        returns_trend = st.checkbox("Returns Trend", value=False)
    with c2v:
        low_ranks_trend = st.checkbox("Ranks Trend", value=True)
        Price_trend = st.checkbox("Price", value=True)
        recommendations_table = st.checkbox("Summary", value=False)

    viz_instructions = []
    if Pie_chart:
        viz_instructions.append("- Industry: Pie Chart of Industries of selected stocks")
    if Return_hold:
        viz_instructions.append("- Expected Returns: line chart for each of the selected stocks with two points for each - first point starting at (0,0) and second point X is number of days to hold (Score_HoldPeriod in high_risk table) vs High Zoltar Rank (y-axis), making starting point for x-axis max(Date) and iterating days forward from that point.")
    if low_ranks_trend:
        viz_instructions.append("- Low Zoltar Rank Over Time: a pretty line chart of Low Zoltar Rank of each stock over time (low_risk table)")
    if recommendations_table:
        viz_instructions.append("- Recommendations: Table of model recommendations for each stock")
    if returns_trend:
        viz_instructions.append("- High Zoltar Rank Over Time: a pretty line chart of High Zoltar Rank of each stock over time (high_risk table)")
    if Price_trend:
        viz_instructions.append("- Price Over Time: a pretty line chart of Close_Price of each stock over time (from high_risk table)")
    viz_section = "\n".join(viz_instructions) if viz_instructions else "- No visualizations selected."
    any_viz = bool(viz_instructions)

    st.sidebar.header("Model Configuration")
    temp_levels = [("0.0 - Exact", 0.0), ("0.1 - Middle", 0.1), ("1.0 - Wild", 1.0)]
    top_p_levels = [("0.5 - Wild", 0.7), ("0.9 - Middle", 0.9), ("1.0 - Exact", 1.0)]
    st.sidebar.write("Temperature setting:")
    temperature = segmented_buttons("Temperature Level", temp_levels, "temp")
    st.sidebar.write("Top-p setting:")
    top_p = segmented_buttons("Top-p Level", top_p_levels, "top_p")

    st.sidebar.markdown(
        """<a href="https://github.com/apod-1/ZoltarFinancial/raw/main/docs/User_Agreement.txt" target="_blank">
        <button class="disclaimer-btn" title="By using this app, you agree to the terms and conditions. This is not investment advice.">View Disclaimer</button></a>""",
        unsafe_allow_html=True)
    with st.expander("Data status", expanded=False):
        for t, n in db_status.items():
            st.write(f"{'✅' if n else '⚠️'} `{t}` — {n if n else 'not loaded (source file not found)'}")
        st.caption(f"DB file: {db_file_used} · app v{APP_VERSION}")

# ======================================================================================
# Bubbles (unchanged)
# ======================================================================================
if show_top_symbols and "low_risk" in DB_TABLES_LOADED:
    try:
        top_symbols = generate_top_10_stream(db_file_used, int(top_n1), int(top_n2))
    except Exception as e:
        print(f"bubbles failed: {e}")
        top_symbols = []
    if top_symbols:
        mid = len(top_symbols) // 2
        with col1:
            st.markdown("<div style='text-align:center; font-size:1em; font-weight:600; color:#b22222; margin-bottom:0.2em;'>"
                        "Top <span style='color:#DAA520;'>Low Zoltar Rank</span> Stocks</div>", unsafe_allow_html=True)
            display_bubbles(col1, top_symbols[:mid])
        with col3:
            st.markdown("<div style='text-align:center; font-size:1em; font-weight:600; color:#b22222; margin-bottom:0.2em;'>"
                        "Top <span style='color:#DAA520;'>High Zoltar Rank</span> Stocks</div>", unsafe_allow_html=True)
            display_bubbles(col3, top_symbols[mid:])

# ======================================================================================
# Prompts (v3.7_F wording kept; tool references updated to the plain tool name)
# ======================================================================================
INSTRUCTION = """You are a helpful chatbot that can interact with an SQL database
for Stock trading education app. You will take the users' questions and turn them into SQL
queries using the tools available. Once you have the information you need, you will
answer the user's question using the data returned.
high risk scores should be communicated as high Zoltar Ranks in context, and low risk scores are low Zoltar Ranks for context.
These scores predict returns - high is for best return in next 14 days, and low is average expected return for the next 14 days.
User is usually interested in high returns, and if stable returns are preferred, low risk scores (low zoltar rank) should be used,
with sorting always done with highest values on top.
If user is interested in ratings, go to ratings_detail and get necessary data (by Symbol). the RatingPublishedAt example format(2025-03-21T11:52:24Z) is not a timestamp format (but can extract timestamp info)

Use the execute_query tool to issue SQLite SELECT queries. To discover schema use
  SELECT name FROM sqlite_master WHERE type='table'   and   PRAGMA table_info(<table>)
When recommending an action, you have to take that action.
Be mindful of space used and limit as much as possible upfront in SQL queries output (always LIMIT, always filter to max(Date) unless asked otherwise).

Available tables (columns):
  high_risk / low_risk / all_high_risk / all_low_risk: Date, Symbol, Score, Score_Sharpe, Score_HoldPeriod, Close_Price, Cap_Size, Sector, Industry, source
  fundamentals: Symbol, Fundamentals_OverallRating, total_ratings, Fundamentals_Sector, Fundamentals_Industry, Fundamentals_Dividends, Fundamentals_PE, Fundamentals_PB, Fundamentals_MarketCap, Fundamentals_avgVolume2Weeks, Fundamentals_avgVolume30Days, Fundamentals_52WeekHigh, Fundamentals_52WeekLow, Fundamentals_52WeekHighDate, Fundamentals_52WeekLowDate, Fundamentals_Float, Fundamentals_SharesOutstanding, Fundamentals_CEO, Fundamentals_NumEmployees, Fundamentals_YearFounded, Fundamentals_ExDividendDate, Fundamentals_PayableDate, Fundamentals_Description
  ratings_detail: Symbol, RatingType, RatingText, RatingPublishedAt
  shap_summary_Large / shap_summary_Mid / shap_summary_Small: Symbol + one REAL column per feature (SHAP values) — check column names with PRAGMA.

SHAP REASONS ARE NOT IN FUNDAMENTALS - they are in the 3 shap_summary tables, joined by Symbol (and Cap_Size tells which one to use); if a symbol is not there it is not in top stocks currently.
all_high_risk and all_low_risk contain intraday production runs of Zoltar Ranks (Date column); high_risk and low_risk contain only daily production runs (but go further back). Unless there is a reason to look at only the most recent intraday data, there is no reason to use 'all' datasets.
When user asks for time-related tasks, Date column should be used in conjunction with Symbol, which represent Tickers, or Stocks.
fundamentals dataset is updated only once a day; all_ datasets contain intraday data; low_risk and high_risk contain daily data.
When user wants the most recent trends, always take the max(Date) for the answer for each Symbol, and all_ files usually provide a better answer. For long-term trends the other ones are used.
The tables are related to each other by Symbol, and additionally by Date if available.
Important: Since many dates are available for same Symbol in high_risk and low_risk data, only the latest date should be used for most queries (unless explicitly stated otherwise)
Always order by descending date first (pick only records with max date unless stated otherwise), then descending Returns for the final answer, and sometimes in order of descending dividends.
When user requests Top stocks, they mean stocks with highest expected returns (highest Zoltar Ranks - low or high, depending on preference).
Ensure final answer meets all criteria set by the user request, and the answer contains non-duplicate symbols that look at the most recent data point, and mention the date used in the answer.
When stocks symbols are presented, also mention current price, and a few ratings/explanations, and when some information is missing, work with the information that is available (fundamentals and SHAP data could be missing).
When the user asks for Top stocks without mentioning High or Low, assume stocks that are in the top 10 for both Low and High Zoltar Ranks are needed.
When user asks for reasons for stocks being selected, refer to SHAP datasets using Cap_Size and Symbol (can check all 3 by Symbol)
When user asks for alpha, the comparison with SPY returns needs to be made.

User prefers the answer in a table format with relevant statistics, and a summary brief.  Response always ends with the phrase 'May the riches be with you...'
"""

AGENT1_SYSTEM = INSTRUCTION + """
Your role is this: You are a database interface. Use the execute_query tool to understand the database/table contents and pull relevant information from the tables,
then answer the user's question by looking up information in the database, running any necessary queries, and responding to the user.
Provide a comprehensive report on each of the selected stocks with data available on the database and provide all final results in text to be used by subsequent agents to summarize further.
If you recommend an action, you must take that action.
"""

AGENT1_SUFFIX = (" ** end of user question** To fully answer this question, after the stock symbols of interest are known, "
                 "limit to top 5 and in your response include information on them from Zoltar Ranks Database fundamentals table "
                 "using the execute_query tool for subsequent agents to use, and include sector, P/E, Dividends, 52Week highs and Lows, Overall Rating. "
                 "Finish with one line exactly like:  SYMBOLS: AAA, BBB, CCC  listing the tickers you selected.")


def parse_symbols(text: str, valid: set) -> list:
    m = re.search(r"SYMBOLS:\s*([A-Z0-9.,\-\s]+)", text or "")
    cands = []
    if m:
        cands = [t.strip().upper() for t in m.group(1).split(",")]
        cands = [t for t in cands if re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,5}", t)]
    if not cands:  # fallback: any known ticker mentioned in the text
        toks = set(re.findall(r"\b[A-Z]{1,5}\b", text or ""))
        cands = [t for t in toks if t in valid]
    out = []
    for s in cands:
        if (not valid or s in valid) and s not in out:
            out.append(s)
    return out[:8]


def df_to_markdown(df: pd.DataFrame) -> str:
    """Minimal GitHub-style table (avoids the `tabulate` dependency DataFrame.to_markdown needs)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v).replace("|", "\\|") for v in row.tolist()) + " |")
    return "\n".join(lines)


def shap_table(symbols: list) -> pd.DataFrame:
    """Deterministic replacement for v3.7_F Agent 5's model-written SHAP SQL. Top-5 |SHAP| per symbol
    across the three cap-size tables; a symbol absent from all three is reported as 'No SHAP data'."""
    rows = []
    for sym in symbols:
        found = False
        for table in SHAP_TABLES:
            if table not in DB_TABLES_LOADED:
                continue
            try:
                df = pd.read_sql_query(f'SELECT * FROM "{table}" WHERE Symbol = ? LIMIT 1', db_conn, params=(sym,))
            except Exception:
                continue
            if df.empty:
                continue
            found = True
            num = df.select_dtypes(include="number").iloc[0].dropna()
            num = num[num != 0]
            for feat, val in num.abs().sort_values(ascending=False).head(5).items():
                v = float(num[feat])
                rows.append({"Symbol": sym, "SHAP Table": table.replace("shap_summary_", ""), "Feature": feat,
                             "SHAP Value": f"{v:.9f}", "Impact": "Increasing" if v > 0 else "Decreasing"})
        if not found:
            rows.append({"Symbol": sym, "SHAP Table": "—", "Feature": "—", "SHAP Value": "—", "Impact": "No SHAP data found"})
    return pd.DataFrame(rows)


# ======================================================================================
# Streaming stage runner
# ======================================================================================
class Stage:
    """One agent stage: st.status box + streaming placeholder + toast, in the v3.7_F style."""

    def __init__(self, key: str, label: str, canvas):
        self.key, self.label = key, label
        with canvas:
            self.status = st.status(f"⏳ {label}", expanded=True)
            with self.status:
                self.events = st.empty()
                self.text_ph = st.empty()
        self.toast = st.toast(label, icon="⏳")
        self.buf, self.ev, self._last = "", [], 0.0

    def on_text(self, delta: str):
        self.buf += delta
        now = time.time()
        if now - self._last > 0.08:
            self.text_ph.markdown(self.buf + " ▌")
            self._last = now

    def on_event(self, kind: str, payload):
        if kind == "tool_call":
            sql = payload.get("args", {}).get("sql")
            self.ev_add(f"🗄️ DB CALL: `{(sql or json.dumps(payload.get('args')))[:160]}`")
        elif kind == "search":
            self.ev_add(f"🔎 Searching: *{payload}*")
        elif kind == "tool_result":
            pass

    def ev_add(self, line: str):
        self.ev.append(line)
        self.events.markdown("\n".join(f"- {e}" for e in self.ev[-8:]))

    def done(self, ok: bool = True, note: str = "", keep_open: bool = False):
        self.text_ph.markdown(self.buf)
        icon = "✅" if ok else "❌"
        self.status.update(label=f"{icon} {self.label}{(' — ' + note) if note else ''}",
                           state="complete" if ok else "error", expanded=keep_open)
        self.toast.toast(self.label, icon=icon)


def run_stage(provider, stage: Stage, system: str, user: str, tools=None, web=None):
    res = provider.run(system=system, user=user, tools=tools, web_search=web,
                       temperature=temperature, top_p=top_p, on_text=stage.on_text, on_event=stage.on_event)
    return res


def run_plot_script(code: str) -> tuple:
    """Execute a model-written plotting script locally. Returns (ok, message)."""
    if os.path.exists(PLOT_PATH):
        os.remove(PLOT_PATH)
    ns = {"pd": pd, "np": np, "plt": plt, "sns": sns, "json": json, "datetime": datetime,
          "query": query_df, "OUTPUT_PATH": PLOT_PATH, "__name__": "__zoltar_plot__"}
    try:
        plt.close("all")
        exec(compile(code, "<agent_plot>", "exec"), ns)  # noqa: S102 — model code, DB is read-only and local
        if not os.path.exists(PLOT_PATH):
            fig = plt.gcf()
            if fig.get_axes():
                fig.savefig(PLOT_PATH, dpi=110, bbox_inches="tight")
        if not os.path.exists(PLOT_PATH):
            return False, "script ran but did not save OUTPUT_PATH"
        with open(PLOT_PATH, "rb") as f:
            b = f.read()
        if is_blank_png(b):
            return False, "saved image is blank"
        st.session_state.image = b
        return True, f"{len(b)} bytes"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1200:]}"
    finally:
        plt.close("all")


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*(.*?)```", text or "", re.S | re.I)
    return m.group(1).strip() if m else (text or "").strip()


PLOT_SYSTEM = """You write self-contained Python plotting scripts. The script will be executed by the host app
with these names ALREADY DEFINED (do not import or redefine them): pd, np, plt, sns, json, datetime,
query(sql) -> pandas.DataFrame (runs a SQLite SELECT on the Zoltar database), OUTPUT_PATH (str).
Rules: no file/network access other than query(); no plt.show(); build ONE landscape figure with the requested
sections side by side (plt.subplots(1, n, figsize=(6*n, 5))); rotate x tick labels -45 degrees; finish with
plt.tight_layout(); plt.savefig(OUTPUT_PATH, dpi=110). Limit Date ranges to the last 3 months in SQL and filter
symbols with WHERE Symbol IN (...). Return ONLY one ```python code block, then a short section titled
'References to visualization' discussing what the chart shows."""

# ======================================================================================
# Main column: query + orchestration
# ======================================================================================
with col2:
    user_query = st.text_input("Your question", value="Best stocks to get now?", label_visibility="collapsed",
                               placeholder="Ask your stock-related question...",
                               help="Ask about best stocks, dividends, sectors, explanations (anything stocks related)")
    go = st.button("Submit Query", type="primary")

    if go:
        prep_db = st.toast("UPDATING ZOLTAR DATABASE...", icon="⏳")
        try:
            with open("agent_repo_t.json", "w") as f:
                json.dump(st.session_state.agent_repo, f)
        except Exception:
            pass
        st.session_state.agent_repo = {"agents": {}, "execution_order": []}
        st.session_state.final_agent_result = ""
        st.session_state.agent_progress = {}
        st.session_state.image = None
        st.session_state.last_run_meta = {"provider": provider_kind, "model": model_name,
                                          "started": datetime.now().isoformat()}

        def add_agent_result(key, data):
            st.session_state.agent_repo["agents"][key] = data
            if key not in st.session_state.agent_repo["execution_order"]:
                st.session_state.agent_repo["execution_order"].append(key)

        def saved(key):
            return st.session_state.agent_repo["agents"].get(key, {}).get("result")

        try:
            provider = make_provider(provider_kind, {"gemini": GOOGLE_API_KEY, "openai": OPENAI_API_KEY}.get(provider_kind, ""),
                                     model_name)
        except Exception as e:
            st.error(f"Could not start the {provider_kind} provider: {e}")
            st.stop()

        if DB_TABLES_MISSING:
            st.warning(f"Tables not loaded (source files not found): {', '.join(DB_TABLES_MISSING)} — answers will be limited.")
        prep_db.toast("UPDATED ZOLTAR DATABASE!!!  ", icon="✅")

        canvas = st.container()
        final_ph = st.empty()
        valid_syms = known_symbols()
        max_attempts = 3

        for attempt in range(1, max_attempts + 1):
            try:
                # ---------------- AGENT 1: Zoltar database ----------------
                if not st.session_state.agent_progress.get("agent1_zoltar"):
                    s1 = Stage("agent1_zoltar", "AGENT 1...ZOLTAR DATABASE", canvas)
                    r1 = run_stage(provider, s1, AGENT1_SYSTEM, user_query + AGENT1_SUFFIX, tools=[EXECUTE_QUERY_TOOL])
                    if r1.error or not r1.text.strip():
                        s1.done(False, r1.error or "empty answer")
                        raise RuntimeError(f"Agent 1: {r1.error or 'empty answer'}")
                    # accuracy check (v3.7_F step 2/3)
                    chk = provider.run(system=INSTRUCTION,
                                       user=user_query + f"""
You are checking work performed by Agent #1, whose task it is to: Understand user query, and construct SQL queries and use available tools to gather information from Zoltar Database for requested Summary of Selected Stocks section.
Here's Agent 1 task and response: {r1.text}
Respond with a single word: ACCURATE or INACCURATE""",
                                       temperature=0.0, top_p=1.0)
                    add_agent_result("agent1_check", {"result": chk.text, "timestamp": datetime.now().isoformat(),
                                                      "source": "Zoltar Database Query Check"})
                    if "INACCURATE" in chk.text.upper():
                        st.toast("INACCURACY IDENTIFIED, RE-PULLING...", icon="❌")
                        s1.ev_add("⚠️ checker flagged INACCURATE — re-running with the checker's note")
                        s1.buf = ""
                        r1b = run_stage(provider, s1, AGENT1_SYSTEM,
                                        user_query + AGENT1_SUFFIX + f"\nA reviewer judged a previous attempt INACCURATE: {chk.text[:500]}. Re-query carefully.",
                                        tools=[EXECUTE_QUERY_TOOL])
                        if not r1b.error and r1b.text.strip():
                            r1 = r1b
                    add_agent_result("agent1_zoltar", {"result": r1.text, "timestamp": datetime.now().isoformat(),
                                                       "source": "Zoltar Database Query", "tool_calls": r1.tool_calls,
                                                       "model": f"{r1.provider}/{r1.model}", "elapsed_s": round(r1.elapsed_s, 1)})
                    st.session_state.agent_progress["agent1_zoltar"] = True
                    s1.done(True, f"{len(r1.tool_calls)} DB calls")
                agent_result = saved("agent1_zoltar")
                symbols = parse_symbols(agent_result, valid_syms)
                st.session_state.last_run_meta["symbols"] = symbols

                # ---------------- AGENT 2: live news & sentiment ----------------
                if not st.session_state.agent_progress.get("agent2_news"):
                    s2 = Stage("agent2_news", "AGENT 2...NEWS ARTICLES", canvas)
                    web = WebSearchSpec(allowed_domains=selected_domains if (strict_domains and selected_domains) else [])
                    msg2 = (f"Search for the latest News and analyze Sentiment for these stocks: {', '.join(symbols) or '(see prior agent result)'}. "
                            f"Use your live web search tool. When searching, only look at the sources specifically selected by the user: {source_str}. "
                            f"Create a table with the best 3 links for detailed reading per stock, and a Sentiment table with columns: "
                            f"Symbol, Analyst Consensus, Blogger Sentiment, Crowd Wisdom, News Sentiment (write 'unknown' for any cell you found no evidence for). "
                            f"Here is the result of the first agent findings: {agent_result[:6000]} ** end of prior agent results** "
                            f"Provide all final results in text to be used by subsequent agents to summarize further.")
                    r2 = run_stage(provider, s2, "You are a financial news and sentiment research analyst. Cite the pages you used.",
                                   msg2, web=web)
                    if r2.error and not r2.text.strip():
                        s2.done(False, r2.error)
                        raise RuntimeError(f"Agent 2: {r2.error}")
                    if r2.citations:
                        s2.ev_add("🔗 " + " · ".join(f"[{c.domain or c.title[:30]}]({c.url})" for c in r2.citations[:8]))
                    add_agent_result("agent2_news", {"result": r2.text, "timestamp": datetime.now().isoformat(),
                                                     "sources": source_str, "search_evidence": r2.search_evidence,
                                                     "citations": [c.__dict__ for c in r2.citations],
                                                     "search_queries": r2.search_queries,
                                                     "model": f"{r2.provider}/{r2.model}", "elapsed_s": round(r2.elapsed_s, 1)})
                    st.session_state.agent_progress["agent2_news"] = True
                    s2.done(True, r2.search_evidence)
                agent_result2 = saved("agent2_news")
                search_evidence = st.session_state.agent_repo["agents"]["agent2_news"].get("search_evidence", "unknown")

                # ---------------- AGENT 3 (+4 fallback): plots ----------------
                if not st.session_state.agent_progress.get("agent3_plots"):
                    s3 = Stage("agent3_plots", "AGENT 3+4...OVERVIEW PLOTS", canvas)
                    if not any_viz or not symbols:
                        s3.buf = "No visualizations selected." if not any_viz else "No symbols identified — plot skipped."
                        add_agent_result("agent3_plots", {"result": s3.buf, "timestamp": datetime.now().isoformat(), "visualizations": viz_section})
                        st.session_state.agent_progress["agent3_plots"] = True
                        s3.done(True, "skipped")
                    else:
                        feedback, ok, r3 = "", False, None
                        for tries in range(1, 4):
                            s3.ev_add(f"🧪 plot attempt #{tries}")
                            s3.buf = ""
                            msg3 = (f"Symbols: {symbols}. Build one figure with these sections:\n{viz_section}\n"
                                    f"Context from Agent 1: {agent_result[:3000]}\n" + (f"\nPrevious attempt failed with: {feedback}\nFix it." if feedback else ""))
                            r3 = run_stage(provider, s3, PLOT_SYSTEM, msg3)
                            if r3.error and not r3.text.strip():
                                feedback = r3.error
                                continue
                            ok, feedback = run_plot_script(extract_code(r3.text))
                            s3.ev_add(("✅ plot saved " if ok else "❌ ") + feedback.splitlines()[0][:200])
                            if ok:
                                break
                        commentary = re.sub(r"```.*?```", "", r3.text if r3 else "", flags=re.S).strip()
                        add_agent_result("agent3_plots", {"result": commentary if ok else f"Plot could not be generated after 3 attempts. Last error: {feedback}",
                                                          "timestamp": datetime.now().isoformat(), "visualizations": viz_section,
                                                          "plot_ok": ok, "model": f"{r3.provider}/{r3.model}" if r3 else ""})
                        st.session_state.agent_progress["agent3_plots"] = True
                        s3.done(ok, "plot ready" if ok else "no plot (report continues)")
                        if ok:
                            with canvas:
                                show_image(st.session_state.image, "Generated Plot")
                agent_result2b = saved("agent3_plots")

                # ---------------- AGENT 5: SHAP ----------------
                if not st.session_state.agent_progress.get("agent5_shap"):
                    s5 = Stage("agent5_shap", "AGENT 5...SHAP ANALYSIS", canvas)
                    shap_df = shap_table(symbols) if symbols else pd.DataFrame()
                    shap_md = df_to_markdown(shap_df) if not shap_df.empty else "No symbols identified — SHAP lookup skipped."
                    s5.ev_add(f"🗄️ SHAP rows: {len(shap_df)} across {', '.join(t for t in SHAP_TABLES if t in DB_TABLES_LOADED) or 'no SHAP tables loaded'}")
                    r5 = run_stage(provider, s5, INSTRUCTION,
                                   f"Here is the SHAP table (top-5 |SHAP| features per symbol, computed from the Zoltar SHAP tables):\n\n{shap_md}\n\n"
                                   f"Write the 'SHAP analysis' section for the final report: reproduce the table as-is, then explain for each stock what "
                                   f"drives its Zoltar Rank (Increasing vs Decreasing). Symbols marked 'No SHAP data found' must be reported as missing, not guessed.")
                    add_agent_result("agent5_shap", {"result": (shap_md + "\n\n" + r5.text) if not r5.error else shap_md,
                                                     "timestamp": datetime.now().isoformat(), "shap_rows": len(shap_df)})
                    st.session_state.agent_progress["agent5_shap"] = True
                    s5.done(not r5.error, f"{len(shap_df)} rows")
                agent_result4 = saved("agent5_shap")

                # ---------------- AGENT 6: compile ----------------
                s6 = Stage("agent6_final", "AGENT 6...COMPILE REPORT", canvas)
                msg6 = f"""Combine the results of prior agents into a comprehensive report, and make sure to use all information synthesized by prior agents to answer this original query: {user_query}. ** End of User Query **
Here is the result of the first agent findings: {agent_result}. ***End of AGENT 1 results***
Here is the result of the second agent findings (live search evidence: {search_evidence}): {agent_result2}. ***End of AGENT 2 results****
And this is commentary of the supporting plots: {agent_result2b} *** End of Agent 3 Results ***
And this is the SHAP section: {agent_result4}  *** End of Agent 4 Results ***
The final report needs to have an executive structure, containing
    1. Summary section with a sentence capturing the essence of the report and table of Fundamentals/About Information and overall recommendation column (Buy, Mixed, Sell),
    2. News and Ratings section with Summary table for News and for Analyst Ratings with columns: Analyst Consensus, Blogger Sentiment, Crowd Wisdom, News Sentiment;
       Make sure to include the links section for each stock listed (from agent 2 results) below the summary table. If the live search evidence says no sources were found, say so explicitly in this section instead of inventing sentiment.
    3. Quant Section with Zoltar Ranks, their direction and SHAP discussion;
    4. Conclusion based on contents of prior section.
Return just the Final Executive Report and nothing else. Response always ends with the phrase 'May the riches be with you...'"""
                r6 = run_stage(provider, s6, INSTRUCTION, msg6)
                if r6.error and not r6.text.strip():
                    s6.done(False, r6.error)
                    raise RuntimeError(f"Agent 6: {r6.error}")
                st.session_state.final_agent_result = r6.text
                add_agent_result("agent6_final_report", {"result": r6.text, "timestamp": datetime.now().isoformat(),
                                                         "source": "Final Executive Report", "model": f"{r6.provider}/{r6.model}"})
                s6.done(True)
                final_ph.markdown("---\n\n" + r6.text)
                st.toast("Final report completed!", icon="✅")
                st.balloons()
                break
            except Exception as e:
                err = st.empty()
                err.error(f"Run failed (attempt {attempt}/{max_attempts}): {e}")
                st.toast("I ran into trouble...RESTARTING", icon="❌")
                time.sleep(1.5)
                err.empty()
                if attempt == max_attempts:
                    st.error("All attempts failed. Please try again with less complex settings (fewer visualizations, cheaper model).")

        try:
            with open("agent_repo.json", "w") as f:
                json.dump(st.session_state.agent_repo, f, default=str)
        except Exception:
            pass

    # ================================================================================
    # Post-run: history + share (unchanged behaviour)
    # ================================================================================
    if not go and st.session_state.final_agent_result:
        if st.session_state.image and not is_blank_png(st.session_state.image):
            show_image(st.session_state.image, "Generated Plot")
        st.markdown("---\n\n" + st.session_state.final_agent_result)

    agent_keys = st.session_state.agent_repo["execution_order"]
    if any(st.session_state.agent_repo["agents"][k].get("result") for k in agent_keys):
        meta = st.session_state.last_run_meta
        with st.expander("Tracking of Agent interactions throughout the run:"):
            st.caption(f"Engine: {meta.get('provider')}/{meta.get('model')} · symbols: {', '.join(meta.get('symbols', []) or ['—'])}")
            st.write("### Execution Sequence")
            for idx, k in enumerate(agent_keys, 1):
                st.write(f"{idx}. **{k}** ({st.session_state.agent_repo['agents'][k]['timestamp']})")
            tabs = st.tabs(agent_keys)
            for tab, k in zip(tabs, agent_keys):
                d = st.session_state.agent_repo["agents"][k]
                with tab:
                    st.markdown(f"#### Agent: `{k}`")
                    st.write(f"**Timestamp:** {d['timestamp']}")
                    st.write("**Raw Result:**")
                    st.code(d.get("result") or "", language="text")
                    st.write("**Metadata:**")
                    st.json({kk: vv for kk, vv in d.items() if kk != "result"})
            if st.button("Load Previous Agent Repository"):
                for fn in ("agent_repo_t.json", "agent_repo.json"):
                    if os.path.exists(fn):
                        with open(fn) as f:
                            st.session_state.agent_repo = json.load(f)
                        st.toast(f"Loaded {fn}", icon="✅")
                        st.rerun()

        with st.popover("✅ Ready to share the results?"):
            from docx import Document
            from docx.shared import Inches
            from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
            import smtplib
            from email import encoders
            from email.mime.base import MIMEBase
            from email.mime.image import MIMEImage
            from email.mime.multipart import MIMEMultipart
            from email.mime.text import MIMEText

            def add_bold_runs(paragraph, text):
                for part in re.split(r"(\*\*.*?\*\*)", text):
                    if part.startswith("**") and part.endswith("**"):
                        paragraph.add_run(part[2:-2]).bold = True
                    else:
                        paragraph.add_run(part)

            def save_to_docx(content, filename, image_path=PLOT_PATH):
                doc = Document()
                doc.add_heading("Your Zoltar Financial Research", level=1).alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
                if os.path.exists(image_path):
                    doc.add_picture(image_path, width=Inches(6))
                for line in content.split("\n"):
                    line = line.strip()
                    if line.startswith("## "):
                        doc.add_heading(line[3:].strip(), level=2).alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
                    elif line.startswith("* ") or line.startswith("- "):
                        add_bold_runs(doc.add_paragraph(style="List Bullet"), line[2:].strip())
                    else:
                        add_bold_runs(doc.add_paragraph(), line)
                doc.save(filename)
                return filename

            @st.cache_data(show_spinner=False, ttl=24 * 3600)
            def _logo_b64():
                try:
                    r = requests.get("https://github.com/apod-1/ZoltarFinancial/raw/main/docs/ZoltarSurf2.png", timeout=15)
                    return base64.b64encode(r.content).decode() if r.status_code == 200 else ""
                except Exception:
                    return ""

            def send_email(sender, password, recipient, doc_path):
                msg = MIMEMultipart()
                msg["From"] = f"Zoltar Financial <{sender}>"
                msg["To"] = recipient
                msg["Subject"] = "Your Zoltar Research Report"
                logo = _logo_b64()
                msg.attach(MIMEText(f"""<html><body><h2>Your Stock Plots</h2><img src="cid:stock_price_plot">
<p>Thank you for using Zoltar Financial Research Assistant. Please find attached the generated report.  Pardon our mess - we are working on improving the user experience.  This is not an investment advice.</p>
<p><img src="data:image/png;base64,{logo}" alt="ZoltarSurf" style="max-width: 600px; width: 30%; height: auto;"></p>
<p>May the riches be with you..</p></body></html>""", "html"))
                if os.path.exists(PLOT_PATH):
                    with open(PLOT_PATH, "rb") as img_file:
                        img = MIMEImage(img_file.read())
                        img.add_header("Content-ID", "<stock_price_plot>")
                        img.add_header("Content-Disposition", "inline", filename=PLOT_PATH)
                        msg.attach(img)
                with open(doc_path, "rb") as attachment:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(doc_path)}")
                msg.attach(part)
                try:
                    server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
                    server.login(sender, password)
                    server.send_message(msg)
                    server.close()
                    return True
                except Exception as e:
                    st.error(f"Failed to send email: {e}")
                    return False

            st.header("Share your research results", help="Save this report as a .docx file and email it to yourself.")
            content = st.session_state.get("final_agent_result") or "The results are empty!"
            sender = _secret("GMAIL", "GMAIL_ACCT", "GMAIL_ACCT")
            password = _secret("GMAIL", "GMAIL_PASS", "GMAIL_PASS")
            with st.form("email_form"):
                recipient = st.text_input("Recipient email address")
                if st.form_submit_button("Send Report"):
                    if not recipient or not sender or not password:
                        st.error("Please fill in all fields (and configure GMAIL secrets).")
                    else:
                        doc_path = save_to_docx(content, f"zoltar_financial_research_report_{datetime.now().strftime('%m%d%y')}.docx")
                        st.success(f"Document saved as {doc_path}")
                        if send_email(sender, password, recipient, doc_path):
                            st.success(f"Report sent successfully to {recipient}!")
            doc_now = save_to_docx(content, "zoltar_financial_report.docx")
            with open(doc_now, "rb") as f:
                st.download_button("Download .docx", f, file_name=os.path.basename(doc_now))
