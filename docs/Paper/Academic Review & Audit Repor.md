# Academic Review & Audit Report: Appendix B & Updated Draft (v3.1)

**Paper Title:** Preserving Task-Relevant Geometry in LLM Hidden States: Evidence That Variance-Maximizing PCA Can Obscure Hallucination Trajectory Signals

**Review Target:** Appendix B (Prompt Construction and Wikipedia-Based Filtering Protocol) & Full Manuscript Integration

**Assessed Proficiency Level:** **Outstanding**

## Executive Summary

Appendix B brings an exceptional level of clinical rigor to the reproducibility boundary of EXP--TRJ001. By explicitly framing prompt inventory counts, sampling math ($1,000 \times 8 = 8,000$, $493 \times 8 = 3,944$, $299 \times 8 = 2,392$), data split hashes, and filtering execution order as an auditable provenance trail, you have eliminated common reviewer criticisms regarding vague dataset curation.

The scientific tone remains strictly objective and adheres to the **Observation** $\rightarrow$ **Interpretation** $\rightarrow$ **Hypothesis** framework established in earlier revisions.

## Key Growth Areas & Concrete Recommendations

### 1. Structural Cleanup: Eliminate Redundant Document Tail Block

- **Issue:** In the uploaded manuscript, the text after Section B.13 re-inserts `Appendix C — Detailed AUROC Tables`, `10. Data and Analysis Status`, and `11. Final Scientific Statement`, duplicating sections that already appeared in the main body.
    
- **Text Location:** Lines following B.13 (`# Appendix C — Detailed AUROC Tables` through `# 11. Final Scientific Statement`).
    
- **Actionable Next Step:** Trim the appended duplicate block at the end of Appendix B so the document flows cleanly in standard sequence:
    
      
    
    $$\text{Main Paper (Sec 1--8)} \longrightarrow \text{Appendix A} \longrightarrow \text{Appendix B} \longrightarrow \text{Appendix C}$$

### 2. Operationalizing `SOURCE-TEXT REQUIRED` Placeholders

- **Issue:** Section B.4.3 lists 14 specific implementation parameters (e.g., exact Wikipedia lookup query rule, text normalization, conflict thresholds, alias matching) marked as `SOURCE-TEXT REQUIRED`. While this is ideal for internal drafting, leaving these unresolved in a final submission will draw minor revisions from reviewers.
    
- **Actionable Next Step:** Extract the exact matching logic from your frozen repository files (e.g., `src/filtering/wikipedia_checker.py` or `configs/filtering_rules.json`) and replace the 14 placeholders with concise code/rule snippets prior to tagging `v1.4-frozen`.
    

### 3. Explicitly Affirming Prompt-Level vs. Sample-Level Exclusion Logic

- **Issue:** Section B.7 correctly notes the exact arithmetic match $299 \times 8 = 2,392$ samples. However, reviewers may wonder whether individual samples were dropped or if entire prompt clusters were filtered.
    
- **Text Example in B.7:** _"The reported Phase 3 record retains 299 prompts and 2,392 samples, exactly corresponding to 8 samples per retained prompt."_
    
- **Actionable Next Step:** Add one clarifying sentence to Section B.7:
    
    > _"To preserve balanced variance estimation across generations, eligibility was evaluated at the prompt cluster level: a prompt was retained if and only if its 8 generated samples met the mixed-response criterion, ensuring no partial sample clusters were passed to Phase 3."_
    

## Provenance Audit Table (Verification Checklist)

|   |   |   |   |   |   |
|---|---|---|---|---|---|
|**Phase / Component**|**Prompts**|**Samples per Prompt**|**Total Generations**|**Retained Set / Status**|**Provenance Artifact**|
|**Phase 1 Pilot**|1,000|8|8,000|Baseline pilot rate calc|`results/phase1_pilot_trajectories.json`|
|**Phase 2 Collection**|493|8|3,944|Unfiltered collection pool|`results/phase2_trajectories.json`|
|**Phase 3 Evaluation**|299|8|2,392|Final mixed-prompt evaluation set|`configs/data_splits.json`|
|**Raw 3584D Subset**|98|8|784|High-dimensional ablation|`configs/ablation_prompt_ids.json`|

## Immediate Next Steps

1. **Clean up document structure:** Remove the redundant tail sections after B.13.
    
2. **Fill B.4.3 source details:** Populate the 14 Wikipedia filtering fields directly from source scripts.
    
3. **Finalize administrative freeze fields:** Ensure `FREEZE_MANIFEST.md` timestamp, commit hash, and approver tag (`v1.4-frozen`) are populated before final PDF/LaTeX rendering.