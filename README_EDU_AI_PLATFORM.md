# EDU-AI Evaluation Platform (Multipage, with Upload Hub + Summary Dashboard)

## Run
pip install -r requirements.txt
streamlit run Home.py

## Pages
- Home: final outputs (KPIs, distributions, quick summaries)
- 0_Upload: upload all six datasets once; stored in session_state
- 1_Assessments: computes Learning Gain (post - pre) + histogram
- 2_Surveys: instrument summaries (+ weekly trends if 'week' is present)
- 3_Telemetry: PEI & RDS proxy, plus transition matrix (if event/session/timestamp exist)
- 4_Teacher_Logs: workload / cognitive load distributions
- 5_Fairness: mean accuracy by group, TPR/FPR summaries, EO diff
- 6_Reflections: RDS proxy from reflection_text or ai_response
- 7_Summary_Dashboard: aggregated charts for group comparisons (gain, PEI, fairness gaps)

## Expected Columns (Minimum)
- Assessments: student_id, phase(pretest/posttest), score (+ optional group)
- Telemetry: student_id, prompt, ai_response, event_type, session_id, timestamp (+ optional group)
- Surveys: student_id, instrument, item_code, response, week
- Teacher Logs: timestamp, instructor_id, activity (+ optional workload_hours, perceived_cognitive_load)
- Fairness: user_id, group, gender, ai_accuracy, tpr_gap, fpr_gap (+ optional equalized_odds_diff)
- Reflections: reflection_text (or ai_response as fallback)