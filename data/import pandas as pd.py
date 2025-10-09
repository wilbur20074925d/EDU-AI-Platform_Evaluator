import pandas as pd
import random

# Load the existing dataset
file_path = "survey_psychometrics_1000.csv"
df = pd.read_csv(file_path)
df.columns = [c.strip().lower() for c in df.columns]

# Example reflection and AI response texts
reflections = [
    "I found this week’s topic quite interesting because it connects theory to real-world practice.",
    "Therefore, I believe I need to review the previous lecture for better understanding.",
    "I justified my choices with evidence from the readings.",
    "So that I can improve, I will plan more study sessions before the next quiz.",
    "However, I noticed that I tend to rush during calculations.",
    "Because the experiment failed, I reflected on my assumptions.",
    "Hence, I should organize my code for better debugging next time."
]

ai_replies = [
    "Good reasoning; you clearly linked your reflection to your learning goals.",
    "Try elaborating on what specific evidence supported your conclusion.",
    "You demonstrated metacognitive awareness—keep explaining your thought process.",
    "Consider connecting your next reflection to feedback from peers or instructors.",
    "Excellent observation; maintaining this habit will deepen your understanding."
]

# Add reflection_text and ai_response columns if missing
if "reflection_text" not in df.columns:
    df["reflection_text"] = [random.choice(reflections) for _ in range(len(df))]
if "ai_response" not in df.columns:
    df["ai_response"] = [random.choice(ai_replies) for _ in range(len(df))]

# Save updated CSV
output_path = "survey_psychometrics_1000.csv"
df.to_csv(output_path, index=False)

print(f"✅ Added 'reflection_text' and 'ai_response' columns.\nSaved as: {output_path}")