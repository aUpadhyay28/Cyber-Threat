"""Record a model file's SHA-256 beside it, so the service refuses a model it was not given.

   python scripts/pin_model.py --model models/latest.joblib

The model is a joblib, and loading one deserialises Python objects: whoever can replace the file can
run code inside the risk service and, more quietly, can decide which logins get blocked. After every
training run, pin the model you intend to deploy and ship the digest with your release - either as
the .sha256 file this writes, or as RISK_MODEL_SHA256 in the service's environment.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import argparse

from riskengine.persist import write_digest

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/latest.joblib")
a = ap.parse_args()

digest = write_digest(a.model)
print(f"{a.model}\n  sha256 {digest}\n  written to {a.model}.sha256")
print(f"\nDeploy with either:\n  RISK_MODEL_SHA256={digest}\nor by shipping the .sha256 file next to the model.")
