import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import is_noise_hallucination

test_cases = [
    (".", 0.5, 0.01, True),
    ("...", 0.6, 0.012, True),
    ("Thank you.", 0.8, 0.02, True),
    ("thanks", 0.7, 0.022, True),
    ("हा", 0.6, 0.015, True),
    ("uhhh", 0.5, 0.02, True),
    ("What are the fees for IT branch?", 2.5, 0.06, False),
    ("DDU ma hostel facility che?", 2.0, 0.05, False),
    ("Raj Mehta ke marks batao", 2.2, 0.07, False),
]

passed = 0
for text, dur, rms, expected in test_cases:
    res = is_noise_hallucination(text, dur, rms)
    status = "PASS" if res == expected else "FAIL"
    if res == expected:
        passed += 1
    print(f"[{status}] '{text}' (dur={dur}s, rms={rms}) -> Expected {expected}, Got {res}")

print(f"\nResult: {passed}/{len(test_cases)} tests passed.")
if passed == len(test_cases):
    print("SUCCESS: Noise hallucination filter working as expected!")
    sys.exit(0)
else:
    sys.exit(1)
