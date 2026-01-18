# CLAUDE.md

## Core Principles
- Do not take things on face value, question everything

### Economic Reasoning First
- Every technical decision must serve the goal: **build an economically viable optimizer**
- When a component underperforms, ask "how do I fix this to capture economic value?" not "should I remove this?"
- Removing functionality means losing signal. Lost signal = lost money.

### Research-Backed Decisions
- Search literature (arxiv, production papers) before implementing non-trivial solutions
- Cite at least one paper or production system where the approach worked
- If you can't find evidence it works, be explicit about the uncertainty

### Measure What Matters
- If you build a model, output its diagnostics (calibration, discrimination, distributions)
- If you can't measure it, you can't improve it
- Never ship a metrics file without the metrics needed to evaluate the thing it describes

## Technical Guardrails

### Do
- Parse PostgreSQL arrays: `{value1,value2}` → extract first numeric value
- Deduplicate views by `log_txnid` before analysis
- Filter zero bids before training
- Use `pd.to_datetime(df['log_dt'], format='ISO8601', utc=True)` for mixed timezones
- Join bids→views on `log_txnid` for win rate (not `len(views)/len(bids)`)
- Fill geo nulls with 'Unknown' before encoding

### Avoid
- `class_weight='balanced'` with extreme class imbalance (<1% positive rate) - destroys probability calibration
- Removing models/features without first attempting to fix them
- Hardcoding feature selections - use data-driven selection
- Outputting predictions without calibration diagnostics

### Prefer
- Bayesian shrinkage toward global rates over raw segment rates (sparse data)
- Simpler models with proper calibration over complex models with poor calibration
- Binary filtering (include/exclude) over continuous confidence scaling when data is sparse

## Working Memory

### CTR Model Miscalibration (Resolved)
- **Symptom**: Model predicted 35% CTR when actual was 0.037%
- **Cause**: `class_weight='balanced'` inflates minority class by ~1000x
- **Fix**: Remove class_weight, implement Bayesian shrinkage toward global CTR
- **Lesson**: With extreme imbalance, class weighting destroys the calibration you need for bidding

### Win Rate Model Miscalibration (Open)
- **Symptom**: ECE=0.176, consistently overestimates by ~1.7x
- **Current state**: Removed from bid formula (diagnostic only)
- **TODO**: Investigate if same class_weight issue; fix and restore to capture market signals

### Bid Floor Clipping (Resolved)
- **Symptom**: 92.67% of bids clipped to $5 floor
- **Cause**: Floor too high relative to expected values
- **Fix**: Lower floor to $2, increase target_margin to 30%

## Reference
- `PIPELINE_SPEC.md` - Full technical specification and code examples
- `EDA_*.md` - Data analysis findings
- `SCRATCHPAD.md` - Feel free to use this to make notes for yourself and your findings. Use it to offload things from working memory, and context into this file for you to come back and refernce it later.