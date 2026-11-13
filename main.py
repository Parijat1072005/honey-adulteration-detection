"""
Unified CLI Entrypoint for Honey Adulteration Detection System.
Commands:
  python main.py --build-data               Extract features & build master datasets
  python main.py --train                    Train & benchmark all 5 modeling stages
  python main.py --figures                  Generate publication-quality figures
  python main.py --predict --image <path>   Run end-to-end inference on a tube photo
  python main.py --app                      Launch interactive Streamlit web dashboard
"""
import sys
import argparse
import subprocess

from build_dataset import build_all_datasets
from train import run_training_pipeline
from src.utils.visualization import generate_all_figures
from predict import HoneyAdulterationPredictor, format_report


def main():
    parser = argparse.ArgumentParser(
        description="AI-Assisted Honey Authenticity & Adulteration Detection (Ninhydrin-Proline Assay)"
    )
    parser.add_argument("--build-data", action="store_true", help="Extract features and compile structured datasets")
    parser.add_argument("--train", action="store_true", help="Train and cross-validate Stage A through Stage E models")
    parser.add_argument("--figures", action="store_true", help="Generate all scientific figures and plots")
    parser.add_argument("--predict", action="store_true", help="Predict adulteration and HAI on a tube image")
    parser.add_argument("--image", "-i", type=str, default=None, help="Path to tube image for prediction")
    parser.add_argument("--a520", "-a", type=float, default=None, help="Optional benchtop measured A520")
    parser.add_argument("--app", action="store_true", help="Launch Streamlit web application")

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    if args.build_data:
        build_all_datasets()

    if args.train:
        run_training_pipeline()

    if args.figures:
        generate_all_figures()

    if args.predict:
        if not args.image:
            print("Error: --image <path> is required when using --predict")
            sys.exit(1)
        predictor = HoneyAdulterationPredictor()
        res = predictor.predict_image(args.image, actual_a520=args.a520)
        print(format_report(res))

    if args.app:
        print("Launching Streamlit Web Application on http://localhost:8501 ...")
        subprocess.run(["streamlit", "run", "app/streamlit_app.py"])


if __name__ == "__main__":
    main()

