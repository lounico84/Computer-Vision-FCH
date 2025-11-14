from config import Settings
from pipeline import run_match_analysis


def main():
    settings = Settings()
    run_match_analysis(settings)


if __name__ == "__main__":
    main()