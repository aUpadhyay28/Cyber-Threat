"""Create a .env file with fresh random secrets, from .env.example.

   python scripts/generate_secrets.py            # writes .env (refuses to overwrite one)
   python scripts/generate_secrets.py --print    # prints the secrets instead of writing a file

Every key is 43 characters from the operating system's cryptographic random source, and all of them
are different, which is what production mode checks. The file is created readable by its owner only.
Never commit .env (it is in .gitignore) and never paste these values into chat, tickets or logs.
"""
import argparse
import os
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SECRETS = ("RISK_API_KEY", "RISK_ANALYST_KEY", "RISK_ADMIN_KEY", "REDIS_PASSWORD", "RISK_HASH_KEY")

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(ROOT / ".env"))
ap.add_argument("--print", action="store_true", help="print instead of writing a file")
ap.add_argument("--force", action="store_true", help="overwrite an existing file (old secrets are lost)")
a = ap.parse_args()

values = {name: secrets.token_urlsafe(32) for name in SECRETS}

if a.print:
    for k, v in values.items():
        print(f"{k}={v}")
    sys.exit(0)

out = Path(a.out)
if out.exists() and not a.force:
    sys.exit(f"{out} already exists. Rotating RISK_HASH_KEY orphans every stored profile, so this will not "
          f"overwrite it; pass --force if you really mean to.")

lines = []
for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
    name = line.split("=", 1)[0]
    lines.append(f"{name}={values[name]}" if name in values and "=" in line and not line.startswith("#") else line)

fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
print(f"wrote {out} with fresh values for: {', '.join(SECRETS)}")
print("Still to fill in: RISK_MODEL_SHA256 (after scripts/pin_model.py).")
print("Give the login backend RISK_API_KEY only. Keep RISK_ADMIN_KEY off application servers.")
