# Customer Success Playbook

## Playbook 1 — High Value + High Risk (the segment that matters most)

**Trigger:** churn_probability >= 0.6 AND (monthly_charges >= 70 OR tenure >= 24).

**SLA:** First call within 4 business hours of being added to queue.

**Steps:**
1. Pull SHAP top-5 drivers from the Copilot before the call.
2. Skim last 3 support tickets and most recent NPS comment.
3. Open the call with empathy on the *specific* recent friction (don't read the SHAP list at them).
4. Make the offer matched to the dominant SHAP driver:
   - Contract Month-to-Month → 1-year contract w/ 15% discount.
   - Tech issues / high tickets → Priority Support + named CSM + 10% discount.
   - Senior + low feature usage → free TechSupport + free OnlineSecurity, no contract change.
5. Confirm acceptance in writing within 24h.
6. Log outcome in CRM; mark cohort for 90-day check-in.

## Playbook 2 — High Value + Low Risk (VIPs)

**Trigger:** monthly_charges >= 70 AND churn_probability < 0.3.

**Goal:** delight, not save. Expand.

**Steps:**
1. Quarterly check-in (email + optional call).
2. Early access to new product features.
3. Anniversary acknowledgment at tenure milestones (12, 24, 36 months).
4. Suggest 1 relevant upgrade per year (do not push).

## Playbook 3 — Low Value + High Risk

**Trigger:** monthly_charges < 50 AND churn_probability >= 0.6.

**Goal:** retain efficiently — never spend more than 2 months of revenue.

**Steps:**
1. Email-only campaign (no human touch).
2. Offer: free OnlineBackup or a single $10 bill credit.
3. If no response in 14 days, drop from save list. Acceptable to lose.

## Playbook 4 — Low Value + Low Risk

**Trigger:** Default segment for everyone else.

**Goal:** monitor only.

**Steps:**
1. Quarterly NPS pulse.
2. Trigger Playbook 1 or 3 only if signals cross the risk threshold.
3. No proactive outreach otherwise — preserves CSM capacity for higher-impact segments.

## Cross-segment guardrails

- If the LLM-suggested offer exceeds the CSM's approval authority, surface "needs manager approval" before sending.
- Never bundle 3+ offers in one outreach — acceptance drops sharply ("offer overload").
- Always include a specific calendar date the offer expires (urgency drives acceptance ~18% higher).
