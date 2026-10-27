# Parzen-Shapley / TRAC — 연합학습 참여자 기여도 측정

연합학습 참여자(클라이언트)의 기여도를 측정하는 두 방법과, 전수 재학습으로 구한 정답
Shapley value(Ground Truth)와의 비교를 담은 저장소다.

| 방법 | 언제 측정하나 | 패키지 |
|---|---|---|
| **Parzen-Shapley** | 학습이 끝난 뒤. frozen task-model 임베딩 위의 class-conditional Parzen utility + balanced permutation Monte Carlo Shapley | `src/parzen_shapley/` |
| **TRAC** | 학습 도중. 서버 root 업데이트와 참여자 업데이트의 cosine 정렬을 EMA로 누적한 점수가 집계 가중치를 겸함 | `src/trac/` |

## 데모 프로그램

```bash
pip install -e ".[demo]"        # numpy, torch, torchvision, timm, matplotlib, Pillow
python -m demo                  # 데스크톱 창 (Tkinter + matplotlib)
python -m demo --export figs --mode replay   # 창 없이 그림만 저장
```

메인 화면(방법 설명·설정) → 로딩 화면(데이터셋별 진행률) → 결과 화면(클라이언트별 기여도, GT 대비 산점도·PCC).
자세한 내용은 [`demo/README.md`](demo/README.md). 데이터(`data/`)는 torchvision MNIST / FashionMNIST / CIFAR-10을
`data/{mnist,fmnist,cifar}`에 내려받아 둔다.

## 저장소 구성

| 경로 | 내용 |
|---|---|
| `src/parzen_shapley/` | Parzen-Shapley: `anchors`(앵커 학습), `models`, `utility`(Parzen utility), `estimators`(exact / MC Shapley), `games`, `pipeline`(설정·데이터셋·앵커·utility 조립) |
| `src/trac/` | TRAC: `scoring`(cosine·EMA·signed 가중치), `fl_round`(한 라운드), `models`, `data`(하이퍼파라미터·파티션·root), `diagnostics`(PCC/AUC) |
| `demo/` | 데모 프로그램 |
| `assets/reference/` | 27개 setting의 클라이언트 파티션과 라벨 override |
| `configs/space_exact_gt_manifest.json` | setting 목록과 Ground Truth Shapley |
| `results/` | 데모가 재생하는 저장 결과: `ground_truth_group_shapley_sweep.json`(GT·Parzen), `fltrust_structure_b128.json`·`fltrust_structure_cw24.json`(TRAC)과 라운드별 궤적 |

---

## Parzen-Shapley

Frozen task-model embedding으로 class-conditional Parzen coalition utility를
만들고, cyclic/reverse balanced permutation Monte Carlo로 client Shapley
value를 추정하는 독립 프로젝트다.

이 코드는 HCDV hierarchy를 사용하지 않는다. 핵심은 다음 두 부분이다.

1. 모델을 coalition마다 재학습하지 않고 embedding상의 Parzen posterior로
   coalition utility를 계산한다.
2. 모든 coalition을 열거하지 않고 balanced permutation path에서 방문한
   prefix coalition만 계산하고 캐시한다.

기존 8-block 구현에 있던 2-additive control은 balanced schedule에서
예측값을 바꾸지 않으면서 query만 조금 늘린다는 것이 이번 실험에서
확인되었다. 따라서 권장 추정기는 더 단순한 `balanced_raw_mc`다.

### 핵심 결과

- 27개 frozen-anchor 설정에서 독립 구현이 기존 Exact Parzen Shapley를
  최대 오차 `3.3e-16`으로 재현했다.
- 10-client shifted, 8 block, 100 seed에서:
  - 일반 MC: Exact-Parzen PCC `0.9445`, 평균 query `628.5/1023`
  - balanced raw MC: Exact-Parzen PCC `0.9935`, 평균 query `425.2/1023`
- 고정 seed direct 실행에서 10-client는 평균 `431.6/1023` coalition만
  직접 계산했다. 기존 control-balanced 예측과 최대 오차는 `5.7e-16`이다.
- 5-client는 전체 coalition이 31개뿐이라 8 block에서 전수 cache가 된다.
  비전수를 유지하려면 1 block이 실용적이며, 100 seed 평균
  Exact-Parzen PCC는 `0.9788`이었다.

Ground Truth Group Shapley와의 PCC는 Monte Carlo 오차뿐 아니라 Parzen utility와
coalition training-accuracy utility 사이의 차이도 포함한다. 특히 10-client
mislabel에서 Exact Parzen 자체의 Ground Truth Group Shapley PCC가 평균 `0.9174`이므로
sampling을 더 늘려도 이 값을 넘어서는 방향으로 자동 개선되지는 않는다.

### 폴더 구조

```text
TaskEmbeddingParzenShapley/
├── artifacts/
│   ├── README.md
│   └── anchors/  # 로컬에서 생성되며 Git에서는 제외
├── assets/reference/  # 27개 client partition과 label override
├── configs/
│   ├── mc_protocol.json
│   ├── space_exact_gt_manifest.json
│   └── ground_truth_group_shapley_protocol.json
├── docs/METHOD_KO.md
├── reports/
│   ├── MONTE_CARLO_COMPARISON_KO.md
│   └── GROUND_TRUTH_GROUP_SHAPLEY_REPORT_KO.md
├── results/
│   ├── sampled_balanced8_predictions.json
│   ├── mc_comparison.json
│   ├── parzen_games.json
│   └── ground_truth_group_shapley_sweep.json
├── scripts/
│   ├── train_anchors.py
│   ├── estimate_sampled.py
│   ├── build_parzen_games.py
│   ├── run_mc_comparison.py
│   └── run_tests.py
├── src/parzen_shapley/
└── tests/
```

510 MB의 frozen checkpoint는 저장소에 포함하지 않는다. Checkpoint를 받은
뒤 `artifacts/anchors/`에 배치한다. Monte Carlo 비교는 저장된
`results/parzen_games.json`만으로 실행할 수 있어 checkpoint가 필요하지
않다.

원 실험의 task-model 구조는 `src/parzen_shapley/models.py`에, 27개
client partition과 label override는 `assets/reference/`에 포함했다. Raw
MNIST, Fashion-MNIST, CIFAR-10 image만 저장소에 포함하지 않는다.

## 실행

환경:

```bash
git clone https://github.com/NamKim-il/parzen-shapley.git
cd parzen-shapley
python -m venv .venv
source .venv/bin/activate
pip install -e .
export PYTHONPATH=src
```
