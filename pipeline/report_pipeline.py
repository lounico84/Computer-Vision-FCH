import subprocess
from pathlib import Path
from config import Settings

# Use a fixed Chrome binary path for deterministic headless PDF export on macOS
CHROME_MAC = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


# Execute the configured notebook, export it to HTML (no code), and render a PDF via headless Chrome
def build_pdf_report_from_notebook(settings: Settings):
    notebook = Path(settings.paths.report_notebook)
    pdf_out = Path(settings.paths.report_pdf)
    pdf_out.parent.mkdir(parents=True, exist_ok=True)

    # Run the notebook and export to self-contained HTML while excluding code cells for a clean report
    subprocess.run(
        [
            "jupyter", "nbconvert",
            "--execute",
            "--ExecutePreprocessor.timeout=900",   # Allow long-running analytics cells to finish
            "--to", "html",
            "--embed-images",                     # Inline plots to avoid external asset dependencies
            "--TemplateExporter.exclude_input=True",  # Hide code while keeping outputs and markdown
            str(notebook),
        ],
        check=True
    )

    # Use the nbconvert HTML output as the intermediate artifact for PDF printing
    html_file = notebook.with_suffix(".html")

    # Render the HTML to PDF using Chrome Headless for consistent layout and typography
    subprocess.run(
        [
            CHROME_MAC,
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            "--print-to-pdf-no-header",          # Remove default header/footer for a report-style PDF
            f"--print-to-pdf={pdf_out}",
            str(html_file),
        ],
        check=True
    )

    # Log the generated PDF location for pipeline traceability
    print(f"[PIPELINE] PDF report generated: {pdf_out}")