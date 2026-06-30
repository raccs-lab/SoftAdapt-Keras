## v0.2.0rc1 (2026-06-30)

## v0.2.0rc0 (2026-06-30)

### BREAKING CHANGE

- Ref #16

### Fix

- **AdaptiveLossCallback**: replace logic that relies on recognizing epoch frequency
- **AdaptiveLossCallback**: several changes throughout to make callback more robust

### Refactor

- **SoftAdaptBase**: remove explicit use of beta in _softmax and use class attribute instead
- **LossWeightedSoftAdapt**: remove duplicate init function

## v0.1.4 (2026-06-23)

## v0.1.3 (2025-12-02)

### Fix

- **adaptive_loss.py**: generalize use of callback
- **callbacks/adaptive_loss.py**: introduce fixes to callbacks for use in keras training

### Refactor

- **.gitignore**: add xml files to gitignore for code coverage

## v0.1.2 (2025-05-07)

## v0.1.0rc0 (2025-05-07)
