# Contribution Measurement for Federated Learning

연합학습 참여자클라이언트의 **기여도**를 측정하는 두 가지 방법에 대한 저장소입니다.

| 방법 | 원리 |
|---|---|
| **Parzen-Shapley** | frozen task-model 임베딩 위의 class-conditional Parzen utility로 coalition 가치를 계산하고, balanced permutation Monte Carlo로 Shapley value를 추정 |
| **TRAC** | 서버 root 업데이트와 참여자 업데이트의 cosine 정렬을 EMA로 누적한 점수가 기여도이자 집계 가중치 |

Ground Truth는 MNIST / Fashion-MNIST / CIFAR-10의 setting(라벨 오염 `mislabel-k`, 분포 편향 `noniid-k`)에서
모든 coalition을 재학습해 구한 exact Shapley value입니다. 데모 프로그램은 두 가지 방법의 결과를 Ground Truth와
PCC로 비교해 보여 줍니다.

## Requirements

Python 3.10 이상. 아래 코드를 통해 패키지 설치와 데이터셋(MNIST / Fashion-MNIST / CIFAR-10, 약 330 MB) 다운로드를 진행합니다.

```bash
pip install -e . && python -m demo
```

- 설치 패키지: `numpy`, `torch`, `torchvision`, `matplotlib`, `Pillow`
- `--download-data`: 데이터셋은 torchvision을 통해 `data/{mnist,fmnist,cifar}` 경로에 다운로드 합니다.

## Folder Structure

```text
.
├── src/parzen_shapley/   # Parzen-Shapley
├── src/trac/             # TRAC
├── demo/                 # 데모 프로그램
├── configs/exact_shapley_ground_truth.json   # Ground Truth Shapley value
└── assets/               # setting별 클라이언트 데이터 파티션과 라벨 override
```

## Reproduce (Demo)

```bash
python -m demo                                                            # 데스크톱 창 실행
python -m demo --export figs --dataset mnist --setting mislabel-4          # 창 없이 그림(PNG)만 figs/에 저장
```

**화면 흐름**: 메인 화면(방법 설명, 데이터셋 1개와 오염 참여자 구성 1개 선택, 계산 방식, TRAC 라운드 수) → 로딩 화면(진행률,
TRAC은 라운드별 PCC) → 결과 화면. 결과 화면은 선택한 setting 하나에 대해 그림 3개를 보여 줍니다.

1. 클라이언트별 기여도 막대그래프 (오염 참여자는 붉은색)
2. Ground Truth Shapley value 대비 산점도와 PCC
3. 측정 시간 비교: Exact Shapley(전수 재학습) 대비 선택한 방법의 소요 시간을 로그 눈금 막대로 그리고 감소율을 표기

**측정 시간의 기준**: Exact Shapley는 모든 coalition을 재학습한 프로세스의 wall time(`configs/exact_shapley_ground_truth.json`의
`timing`), Parzen-Shapley는 앵커 학습·임베딩 추출·Parzen 인덱스 구성·추정의 합, TRAC은 FL 학습 전체의 wall clock(3 seed 평균)입니다.
실제 계산 모드에서는 그 자리에서 측정한 시간을 쓰며, TRAC은 실행한 라운드 수를 함께 표기합니다. 저장된 결과를 재생할 때 CIFAR-10의
TRAC은 40라운드 실행 시간을 30라운드로 환산합니다. 세 파일 모두 `timing = {total_seconds, stages, rounds, seeds, source}` 형식을 공유합니다.

**계산 방식**

| 방식 | Parzen-Shapley | TRAC |
|---|---|---|
| 저장된 결과 재생 (기본) | `results/parzen_shapley.json` | `results/trac.json` + `results/trac_trajectories/` |
| 실제 계산 | setting당 20~40초 (첫 실행 시 앵커 모델 학습 포함, GPU 기준). 결과는 `artifacts/demo_cache/`에 캐시 | 라운드당 MNIST/FMNIST 약 19초, CIFAR 약 8초 (RTX 3080 기준). 기본 5라운드 |

실제 계산한 TRAC은 같은 라운드 수의 저장 궤적과 비트 단위로 일치합니다. Parzen-Shapley는 앵커 모델을 다시 학습하므로
저장값과 약 1e-4(CIFAR 약 3e-3) 차이가 납니다.

**옵션** (`--export` 사용 시 `--dataset`과 `--setting`으로 setting을 고릅니다)

| 옵션 | 설명 |
|---|---|
| `--export DIR` | 창 대신 그림을 DIR에 저장 (`{method}_bar.png`, `{method}_scatter.png`, `{method}_time.png`, `schematic.png`) |
| `--dataset {mnist,fmnist,cifar}` | 데이터셋 (기본 mnist) |
| `--setting mislabel-k \| noniid-k` | 오염 참여자 구성. MNIST/FMNIST는 k ∈ {2,4,6,8}, CIFAR-10은 k ∈ {1,2,3,4} (기본값은 데이터셋별 기본 setting) |
| `--mode {replay,real}` | 저장된 결과 재생(기본) / 실제 계산 |
| `--method {both,parzen,trac}` | 실행할 방법 (기본 both) |
| `--rounds N` | 실제 계산 TRAC 라운드 수 (기본 5, 최대 40) |
| `--device {cuda,cpu}` | 계산 장치 (기본 자동 감지) |
| `--download-data` | 데이터셋만 내려받고 종료 |

한글은 Tk 대신 PIL이 그리며, 폰트는 `demo/fonts/`에 포함된 Pretendard(SIL Open Font License)를 사용합니다.
