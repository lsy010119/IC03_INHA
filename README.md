# IC03_INHA
인공지능 특화 연구실 최종 산출물 통합을 위한 인하대 연구 산출물
---

## AoAEstimator

- 계측치 입력으로 바람 속도 추정, 이로부터 유효 받음각을 추정하는 칼만 필터 기반 추정 알고리즘

### Structure Overview

```
AoAEstimator/
├── __init__.py
├── base.py          # Base libraries & Parameters
├── models.py        # Aerodynamic coefficient table approximation model
├── utils.py         # Filter dynamics / measurement model & math utilities
└── aoaestimator.py  # Main estimator (AoAEstimator)

```

### I/O Interface

- 메인 모듈은 `AoAEstimator/aoaestimator.py`의 `AoAEstimator`
- `AoAEstimator.update` 매서드로 입력된 계측치를 바탕으로 상태 변수 전파 및 갱신

**Inputs**
| Parameter | Type | Unit | Shape | Description |
| --- | --- | --- | --- | --- |
| `vel_body` | `np.ndarray` | $\mathrm{m/s}$ | (3,) | 동체 좌표계 기준 속도 |
| `acc_specific` | `np.ndarray` | $\mathrm{m/s^2}$ | (3,) | 동체 좌표계 기준 특이 가속도 |
| `finout` | `np.ndarray` | $\mathrm{rad}$ | (4,) | 조종면(Fin) 편각 |
| `cI_B` | `np.ndarray` | - | (3, 3) | 관성 좌표계 -> 동체 좌표계 DCM |
| `w_body` | `np.ndarray` | $\mathrm{rad/s}$ | (3,) | 동체 각속도 |

> Glide Phase(추력 없이 공력만 작용) 상황을 가정

**Outputs**
- `AoAEstimator.get_state()`를 통해 `np.ndarray` 상태 변수 반환
- 유동 속도(동체 속도 - 바람 속도)로부터 바람이 고려된 유효 받음각 산출
  
| Variable | Type | Unit | Shape | Description |
| --- | --- | --- | --- | --- |
| `x[0:3]` | `np.ndarray` | $\mathrm{m/s}$ | (3,) | 동체 좌표계 기준 속도 추정치 |
| `x[3:6]` | `np.ndarray` | $\mathrm{m/s}$ | (3,) | 동체 좌표계 기준 바람 속도 추정치 |


## LTEstimator

- 트랜스포머 기반 장기 시계열 예측 모델


