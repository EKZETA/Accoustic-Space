# Accoustic-Space

Current deepfake audio detectors focus primarily on vocal artifacts (robotic tones or
unusual inflections). Modern generative AI easily bypasses these biometric checks, making standard audio
fraud detection obsolete.
A security analyst at Infotact uploads a suspected deepfake audio clip. AcousticSpace doesn't
just listen to the voice; it mathematically isolates the background "Room Impulse Response" (how sound
bounces off the walls) and the speaker's breathing patterns. The dashboard instantly flags the audio as
artificially generated because the acoustic reflection of the generated voice does not match the alleged
background environment.

Key Modules:
• Audio Processing Pipeline (Python & Librosa): Extracts low-level acoustic features, specifically
isolating RIR and environmental reverb.
• Transformer Classifier (PyTorch & HuggingFace): A fine-tuned Audio Spectrogram Transformer
(AST) trained to detect mismatches between vocal cadence and spatial acoustics.
• API Gateway (FastAPI): Serves the ML model for low-latency, real-time inference.
• Analyst Dashboard (React): A frontend UI for uploading audio files and visualizing waveform
anomalies and confidence scores.