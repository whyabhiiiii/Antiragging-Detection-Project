from transformers import pipeline

print("Loading model...")
classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-3")
labels = ["Insult or Harassment", "Direct Threat", "Academic Hierarchy Abuse or Mild Ragging", "Severe Ragging or Hazing", "Suicide or Self-harm Risk"]
print("Testing...")
res = classifier("You should just disappear.", labels, multi_label=True)
print(res)
