from config import Settings
from pipeline import run_match_analysis

# Create Settings object and runs match analysis pipeline
def main():
    settings = Settings()
    run_match_analysis(settings)


if __name__ == "__main__":
    main()