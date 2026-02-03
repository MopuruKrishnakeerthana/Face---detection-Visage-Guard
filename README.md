# Anti-Spoofing / Liveliness Detection Project

## Steps to run

1. Install dependencies:
   pip install -r requirements.txt

2. Collect data:
   python collect_data.py
   - Press 'r' for Real, 's' for Spoof, 'q' to quit

3. Train model:
   python train.py
   - Model saved to models/spoof_detector.h5

4. Run live demo:
   python run_realtime.py
   - Press 'q' to quit
