# Collision Check Log — Phase 2 Type A prompts (plan v1.4 §5.4)

Run (UTC): 2026-09-13T14:30:40.249122+00:00

## Method

Same as the Phase 1 batch (`results/collision_check_log.md`): Wikipedia API only, `action=query`, `redirects=1`, batched <=50 titles/request. No second search-API pass (T2). Residual risk statement applies identically -- see the Phase 1 log.

**Total clean prompts kept: 493**

## Per-domain results

| domain | checked | collisions | kept |
|---|---|---|---|
| fictional_movie | 100 | 2 | 98 |
| fictional_person | 100 | 0 | 100 |
| fictional_paper | 100 | 0 | 100 |
| fictional_chemical | 100 | 1 | 99 |
| fictional_location | 100 | 4 | 96 |

## Dropped (collided) candidates

- Golden Threshold
- Last Horizon
- Chlorophenolate
- Dunmere
- Tarndale
- Calderbrook
- Orridge
