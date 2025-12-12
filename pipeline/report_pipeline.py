import subprocess
from pathlib import Path
from config import Settings

CHROME_MAC = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

def build_pdf_report_from_notebook(settings: Settings):
    notebook = Path(settings.paths.report_notebook)
    pdf_out = Path(settings.paths.report_pdf)
    pdf_out.parent.mkdir(parents=True, exist_ok=True)

    # 1) Notebook ausführen + HTML erzeugen
    #    - KEIN Code im Export
    #    - Outputs (Plots, Tabellen, Markdown) bleiben sichtbar
    subprocess.run(
        [
            "jupyter", "nbconvert",
            "--execute",
            "--ExecutePreprocessor.timeout=900",
            "--to", "html",
            "--embed-images",
            "--TemplateExporter.exclude_input=True",
            str(notebook),
        ],
        check=True
    )

    html_file = notebook.with_suffix(".html")

    # 2) HTML → PDF (Chrome Headless, macOS)
    subprocess.run(
        [
            CHROME_MAC,
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            "--print-to-pdf-no-header",
            f"--print-to-pdf={pdf_out}",
            str(html_file),
        ],
        check=True
    )

    print(f"[PIPELINE] PDF report generated: {pdf_out}")