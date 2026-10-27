# Contribution Measurement for Federated Learning

연합학습(Federated Learning) 참여자(클라이언트)의 **기여도**를 측정하는 두 방법과, 전수 재학습으로 구한
정답 Shapley value(Ground Truth)와의 비교를 담은 저장소입니다.

| 방법 | 측정 시점 | 원리 | 패키지 |
|---|---|---|---|
| **Parzen-Shapley** | 학습이 끝난 뒤 | frozen task-model 임베딩 위의 class-conditional Parzen utility로 coalition 가치를 계산하고, balanced permutation Monte Carlo로 Shapley value를 추정 | `src/parzen_shapley/` |
| **TRAC** | 학습 도중 | 서버 root 업데이트와 참여자 업데이트의 cosine 정렬을 EMA로 누적한 점수가 기여도이자 집계 가중치 | `src/trac/` |

Ground Truth는 MNIST / Fashion-MNIST / CIFAR-10의 24개 setting(라벨 오염 `mislabel-k`, 분포 편향 `noniid-k`)에서
모든 coalition을 재학습해 구한 exact Shapley value입니다. 데모 프로그램이 두 방법의 결과를 Ground Truth와
PCC로 비교해 보여 줍니다.

## Requirements

Python 3.10 이상. 아래 한 줄로 패키지 설치와 데이터셋(MNIST / Fashion-MNIST / CIFAR-10, 약 330 MB) 다운로드가 끝납니다.

```bash
pip install -e . && python -m demo --download-data
```

- 설치되는 패키지: `numpy`, `torch`, `torchvision`, `matplotlib`, `Pillow`
- 데이터셋은 torchvision을 통해 `data/{mnist,fmnist,cifar}`에 내려받습니다. 저장된 결과를 재생하는 기본 모드에는
  필요 없고, 실제 계산 모드에서만 쓰입니다. 다운로드를 건너뛰어도 실제 계산 시 자동으로 받습니다.
- 실제 계산 모드는 CUDA GPU를 권장합니다(없으면 CPU로 동작하되 느립니다). 데스크톱 창은 Tkinter를 사용합니다.

## Folder Structure

```text
.
├── src/parzen_shapley/   # Parzen-Shapley: 앵커 학습, 임베딩, Parzen utility, Shapley 추정
├── src/trac/             # TRAC: 라운드 실행, cosine·EMA 점수, 집계, 데이터 파티션
├── demo/                 # 데모 프로그램 (Tkinter + matplotlib), python -m demo
├── configs/exact_shapley_ground_truth.json   # 24개 setting 정의와 Ground Truth Shapley value
├── assets/reference/     # setting별 클라이언트 데이터 파티션과 라벨 override (인덱스만, 이미지 없음)
├── results/
│   ├── parzen_shapley.json        # Parzen-Shapley 결과
│   ├── trac.json                  # TRAC 결과
│   └── trac_trajectories/         # TRAC 라운드별 점수 궤적
├── artifacts/            # 로컬 생성물 (앵커 체크포인트, 데모 캐시) — Git 제외
└── data/                 # 다운로드한 데이터셋 — Git 제외
```

## Reproduce (Demo)

```bash
python -m demo                                  # 데스크톱 창 실행
python -m demo --export figs --mode replay      # 창 없이 그림(PNG)만 figs/에 저장
```

**화면 흐름**: 메인 화면(방법 설명, 데이터셋별 오염 참여자 구성 선택, 계산 방식 선택) → 로딩 화면(데이터셋별 진행률,
TRAC은 라운드별 PCC) → 결과 화면(클라이언트별 기여도 막대그래프, 오염 참여자는 붉은색 / Ground Truth 대비 산점도와 PCC).

**계산 방식**

| 방식 | Parzen-Shapley | TRAC |
|---|---|---|
| 저장된 결과 재생 (기본) | `results/parzen_shapley.json` | `results/trac.json` + `results/trac_trajectories/` |
| 실제 계산 | 데이터셋당 20~40초 (첫 실행 시 앵커 모델 학습 포함, GPU 기준). 결과는 `artifacts/demo_cache/`에 캐시 | 라운드당 MNIST/FMNIST 약 19초, CIFAR 약 8초 (RTX 3080 기준). 기본 5라운드 |

실제 계산한 TRAC은 같은 라운드 수의 저장 궤적과 비트 단위로 일치합니다. Parzen-Shapley는 앵커 모델을 다시 학습하므로
저장값과 약 1e-4(CIFAR 약 3e-3) 차이가 납니다.

**옵션**

| 옵션 | 설명 |
|---|---|
| `--mode {replay,real}` | 저장된 결과 재생(기본) / 실제 계산 |
| `--method {both,parzen,trac}` | 실행할 방법 (기본 both) |
| `--rounds N` | TRAC 라운드 수 (기본 5, 최대 40) |
| `--device {cuda,cpu}` | 계산 장치 (기본 자동 감지) |
| `--export DIR` | 창 대신 그림을 DIR에 저장 |
| `--download-data` | 데이터셋만 내려받고 종료 |

한글은 Tk 대신 PIL이 그리며, 폰트는 `demo/fonts/`에 포함된 Pretendard(SIL Open Font License)를 사용합니다.
