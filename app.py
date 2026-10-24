
!pip -q install -U gradio requests pillow numpy sympy

import gradio as gr
import requests
import numpy as np
import sympy as sp
import json
import base64
import re
from PIL import Image
from io import BytesIO
from datetime import datetime

API="https://quoctk31-mathsnap-ai.hf.space/predict"
FILE="/content/mathink_corrections.jsonl"
FAIL="Expression cannot be evaluated"

CSS="""
body,.gradio-container{
    background:#0b0b0b!important;
    color:#f5f5f5!important;
}
.gradio-container{
    width:98%!important;
    max-width:1800px!important;
    margin:0 auto!important;
    padding:24px!important;
}
.gr-block,.gr-panel,.form{
    background:#252525!important;
    border-color:#393939!important;
}
label,.prose h1,.prose h2,.prose p{
    color:#f5f5f5!important;
}
textarea,input{
    background:#383f4e!important;
    color:#f5f5f5!important;
    border:1px solid #454b59!important;
    font-size:17px!important;
}
button{
    min-height:48px!important;
    font-size:17px!important;
    border-radius:8px!important;
}
button.primary{
    background:#f45108!important;
    border-color:#f45108!important;
    color:white!important;
}
"""

def clean_latex(s):
    s=str(s).strip()
    s=re.sub(r"\s+"," ",s)

    # Normalize spaces inside exponent and subscript braces
    s=re.sub(r"\^\s*\{\s*([^{}]+?)\s*\}",r"^{\1}",s)
    s=re.sub(r"_\s*\{\s*([^{}]+?)\s*\}",r"_{\1}",s)

    # Join coefficient and variable: 3 x -> 3x
    s=re.sub(r"(?<=\d)\s+(?=[a-zA-Z])","",s)

    # Normalize differential: d x -> dx
    s=re.sub(r"\bd\s+([a-zA-Z])\b",r"d\1",s)

    # Join separated digits: 2 0 -> 20
    s=re.sub(r"(?<=\d)\s+(?=\d)","",s)

    # Remove spaces around arithmetic operators
    s=re.sub(r"\s*([=+\-*/])\s*",r"\1",s)

    # Common OCR corrections in likely numeric contexts
    s=re.sub(r"(?<=\d)O(?=\d|\b)","0",s)
    s=re.sub(r"\bO(?=\s*(?:=|[+\-*/)]|$))","0",s)
    s=re.sub(r"(?<![a-zA-Z])q(?=\s*(?:=|[+\-*/)]|$))","9",s)
    s=re.sub(r"(?<=[=+\-*/(])\s*q\b","9",s)

    return s.strip()

def recognize(img):
    if img is None:
        return "","Draw an expression first."
    try:
        if isinstance(img,dict):
            img=img.get("composite")
        if img is None:
            return "","No drawing received."

        im=Image.fromarray(np.asarray(img).astype("uint8")).convert("RGB")
        buf=BytesIO()
        im.save(buf,format="PNG")

        r=requests.post(
            API,
            files={"file":("equation.png",buf.getvalue(),"image/png")},
            timeout=120
        )
        r.raise_for_status()
        latex=r.json().get("latex","")
        if not latex:
            return "","Recognition failed."

        return clean_latex(latex),"Recognition complete. Check the prediction."
    except Exception:
        return "","Recognition service unavailable. Try again."

def convert(s):
    s=clean_latex(s)
    s=s.replace(r"\left","").replace(r"\right","")
    s=s.replace(r"\times","*").replace(r"\cdot","*").replace(r"\div","/")
    s=s.replace(r"\pi","pi")
    s=re.sub(r"\\(?:,|;|!|quad|qquad)","",s)
    s=re.sub(r"\\(?:mathrm|text)\{([^{}]*)\}",r"\1",s)
    s=re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}",r"((\1)/(\2))",s)
    s=re.sub(r"\\sqrt\{([^{}]+)\}",r"sqrt(\1)",s)
    s=s.replace("^","**").replace("{","(").replace("}",")")

    # Implicit multiplication: 3x -> 3*x, x2 -> x*2
    s=re.sub(r"(?<=\d)(?=[a-zA-Z])","*",s)
    s=re.sub(r"(?<=\))(?=[a-zA-Z0-9(])","*",s)
    s=re.sub(r"(?<=[a-zA-Z])(?=\d)","*",s)
    return s

def evaluate_integral(original):
    s=str(original).strip()
    s=re.sub(r"\s+"," ",s)
    s=re.sub(r"\^\s*\{\s*([^{}]+?)\s*\}",r"^{\1}",s)
    s=re.sub(r"_\s*\{\s*([^{}]+?)\s*\}",r"_{\1}",s)
    s=re.sub(r"\bd\s+([a-zA-Z])\b",r"d\1",s)

    m=re.fullmatch(
        r"\\int(?:_\{?([^{}\s]+)\}?)?(?:\^\{?([^{}\s]+)\}?)?\s*(.*?)\s+d([a-zA-Z])",
        s
    )
    if not m:
        return FAIL

    lo,hi,body,var=m.groups()
    x=sp.Symbol(var)
    body=convert(body)

    if not re.fullmatch(r"[0-9a-zA-Z_+\-*/(). ]+",body):
        return FAIL

    local={"sqrt":sp.sqrt,"pi":sp.pi,"E":sp.E}
    f=sp.sympify(body,locals=local)

    if lo is not None and hi is not None:
        lo=sp.sympify(convert(lo),locals=local)
        hi=sp.sympify(convert(hi),locals=local)
        ans=sp.integrate(f,(x,lo,hi))
        if ans.has(sp.Integral,sp.zoo,sp.nan,sp.oo,-sp.oo):
            return FAIL
        return str(ans)

    if lo is None and hi is None:
        ans=sp.integrate(f,x)
        if ans.has(sp.Integral,sp.zoo,sp.nan,sp.oo,-sp.oo):
            return FAIL
        return str(ans)+" + C"

    return FAIL

def evaluate(latex,values):
    try:
        if not latex.strip():
            return FAIL

        original=clean_latex(latex)
        local={
            "sqrt":sp.sqrt,"sin":sp.sin,"cos":sp.cos,
            "tan":sp.tan,"log":sp.log,"pi":sp.pi,"E":sp.E
        }

        if r"\int" in original:
            return evaluate_integral(original)

        if any(k in original for k in (r"\sum",r"\lim",r"\prod")):
            return FAIL

        s=convert(original)
        if not re.fullmatch(r"[0-9a-zA-Z_+\-*/().=, \t]+",s):
            return FAIL

        subs={}
        if values.strip():
            for item in values.split(","):
                a=item.split("=",1)
                if len(a)!=2 or not a[0].strip().isidentifier():
                    return FAIL
                subs[sp.Symbol(a[0].strip())]=sp.sympify(a[1].strip(),locals=local)

        if "=" in s:
            p=s.split("=")
            if len(p)!=2:
                return FAIL

            left=sp.sympify(p[0],locals=local)
            right=sp.sympify(p[1],locals=local)
            expr=sp.Eq(left,right).subs(subs)

            if expr is sp.true:
                return "True"
            if expr is sp.false:
                return FAIL

            symbols=sorted(expr.free_symbols,key=str)
            if not symbols:
                return "True" if bool(expr) else FAIL

            ans=sp.solve(expr,symbols,dict=True)
            if not ans:
                return FAIL

            out=[]
            for sol in ans:
                parts=[]
                for x in symbols:
                    if x in sol:
                        v=sol[x]
                        if v.has(sp.zoo,sp.nan,sp.oo,-sp.oo):
                            return FAIL
                        parts.append(str(x)+" = "+str(v))
                if parts:
                    out.append(", ".join(parts))
            return "; ".join(out) if out else FAIL

        expr=sp.sympify(s,locals=local).subs(subs)
        if expr.free_symbols:
            return FAIL

        ans=sp.simplify(expr)
        if ans.has(sp.zoo,sp.nan,sp.oo,-sp.oo) or ans.is_finite is False:
            return FAIL
        return str(ans)

    except Exception:
        return FAIL

def preview(latex):
    if not latex.strip():
        return "Enter an expression first."
    return "### Mathematical expression\n\n$$\n"+latex+"\n$$"

def save_correction(img,original,corrected):
    if img is None or not original.strip() or not corrected.strip():
        return "Draw an expression and provide both LaTeX values."
    try:
        if isinstance(img,dict):
            img=img.get("composite")

        im=Image.fromarray(np.asarray(img).astype("uint8")).convert("RGB")
        buf=BytesIO()
        im.save(buf,format="PNG")

        record={
            "time":datetime.now().isoformat(),
            "original_latex":original,
            "corrected_latex":corrected,
            "image_png_base64":base64.b64encode(buf.getvalue()).decode("ascii")
        }

        with open(FILE,"a",encoding="utf-8") as f:
            f.write(json.dumps(record,ensure_ascii=False)+"\n")
        return "Correction saved."
    except Exception:
        return "Could not save correction."

with gr.Blocks(
    title="MathInk AI",
    css=CSS,
    theme=gr.themes.Base(
        primary_hue="orange",
        secondary_hue="slate",
        neutral_hue="slate"
    )
) as demo:

    gr.Markdown("# MathInk AI")
    gr.Markdown("Draw a handwritten equation, recognize it, then edit and evaluate it.")

    drawing=gr.Sketchpad(
        type="numpy",
        image_mode="RGB",
        canvas_size=(1200,400),
        label="Draw your expression"
    )

    with gr.Row():
        run=gr.Button("Recognize",variant="primary")
        clear=gr.ClearButton([drawing],value="Clear")

    status=gr.Textbox(label="Recognition status",interactive=False)
    original=gr.Textbox(label="Original prediction",lines=2)
    corrected=gr.Textbox(
        label="Edit/correct LaTeX here",
        lines=2,
        placeholder=r"Example: x^{2}+y^{2}=7"
    )

    with gr.Row():
        show=gr.Button("Use original prediction")
        render=gr.Button("Check corrected LaTeX")
        save=gr.Button("Save correction",variant="primary")

    preview_box=gr.Markdown("Your expression preview will appear here.")

    gr.Markdown("## Evaluate expression")
    values=gr.Textbox(
        label="Variable values (optional)",
        placeholder="x=2,y=3"
    )
    calc=gr.Button("Evaluate Expression",variant="primary")
    result=gr.Textbox(label="Final answer",lines=3,interactive=False)
    save_status=gr.Textbox(label="Correction status",interactive=False)

    run.click(recognize,inputs=drawing,outputs=[original,status])   
    show.click(lambda x:x,inputs=original,outputs=corrected)
    render.click(preview,inputs=corrected,outputs=preview_box)
    calc.click(evaluate,inputs=[corrected,values],outputs=result)
    save.click(
        save_correction,
        inputs=[drawing,original,corrected],
        outputs=save_status
    )

demo.launch(share=True)
