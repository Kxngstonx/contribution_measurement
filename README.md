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

**화면 흐름**: 메인 화면(방법 설명, 데이터셋과 환경 선택, 계산 방식, TRAC 라운드 수) → 로딩 화면(진행률,
TRAC은 라운드별 PCC) → 결과 화면. 결과 화면은 선택한 환경에 대해 결과를 시각화합니다.

1. 클라이언트별 기여도 막대그래프
2. Ground Truth 대비 산점도와 PCC
3. 측정 시간 비교: Ground Truth와 선택한 방법의 소요 시간을 로그 눈금 막대로 그리고, 방법의 소요 시간이 Ground Truth의 몇 %인지 표기
