from transformers import WhisperForConditionalGeneration
model = WhisperForConditionalGeneration.from_pretrained("KBLab/kb-whisper-large")
model.save_pretrained("/tmp/kb-whisper-large")