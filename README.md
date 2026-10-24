\# MathInk AI



An AI-powered handwritten mathematical expression recognition and evaluation web application.



\## Overview



MathInk AI allows users to draw mathematical expressions on a digital canvas. The application recognizes the handwriting, converts it into LaTeX, allows users to correct the predicted expression, and evaluates supported mathematical expressions.



\## Features



\- Draw handwritten mathematical expressions

\- Recognize expressions using the MathSnap AI API

\- Display and edit predicted LaTeX

\- Preview mathematical expressions

\- Evaluate supported arithmetic expressions and equations

\- Evaluate selected definite and indefinite integrals

\- Enter values for variables

\- Save corrected expressions for reference



\## Technologies Used



\- Python

\- Gradio

\- SymPy

\- NumPy

\- Pillow

\- Requests

\- MathSnap AI recognition API



\## How It Works



1\. The user draws a mathematical expression on the canvas.

2\. The drawing is sent to the MathSnap AI API.

3\. The API returns a predicted LaTeX expression.

4\. The user can review and correct the prediction.

5\. SymPy evaluates supported expressions or solves supported equations.

6\. Corrections can be saved locally as JSONL records.



\## Installation



Install Python, then install the dependencies:



```bash

pip install -r requirements.txt

```



\## Run the Application



```bash

python app.py

```



\## Limitations



\- Handwriting recognition requires internet access and an available MathSnap API.

\- Recognition may produce incorrect predictions.

\- Mathematical evaluation supports a subset of mathematical notation.

\- Saved corrections do not automatically retrain the recognition model.



\## Future Improvements



\- Improve handwriting recognition accuracy.

\- Support more mathematical notation.

\- Use saved corrections to build a training dataset.

\- Add a more comprehensive evaluation engine.



\## Author



MathInk AI Project

