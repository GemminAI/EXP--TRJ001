# Collision Check Log — Type A prompts (plan v1.3 §5.3, T2)

Run (UTC): 2026-09-11T22:34:05.059729+00:00

## Method

Wikipedia API only (`action=query`, `redirects=1`, batched <=50 titles/request). A candidate is dropped if its exact title resolves to any existing Wikipedia page (including via redirect). **No second search-API pass was performed** (T2: no existing Google Custom Search or equivalent implementation was found in the organization; adding one was judged out of scope for this experiment).

## Residual risk (do not read as "100% verified")

This check only catches collisions Wikipedia's own title/redirect resolution surfaces. It will not catch: real entities absent from English Wikipedia (minor place names, non-English-language notable figures, unindexed/very recent works), near-miss title variants not covered by redirects, or entities documented only on other sites. This residual risk is accepted per plan v1.3 §5.3 and is not resolved by this log -- it is recorded here for the record.

## Per-domain results

| domain | checked | collisions | kept |
|---|---|---|---|
| fictional_movie | 220 | 3 | 200 |
| fictional_person | 220 | 0 | 200 |
| fictional_paper | 220 | 0 | 200 |
| fictional_chemical | 220 | 4 | 200 |
| fictional_location | 220 | 0 | 200 |

## Dropped (collided) candidates

- Midnight Garden
- Midnight Shadow
- Deep Harbor
- Vinylsulfonate
- Carbonitrile
- Ethoxzolamide
- Nitrocarbamide
