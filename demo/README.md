# 기여도 평가 데모 프로그램

연합학습 참여자 기여도를 측정하는 두 방법 — **Parzen-Shapley**(학습 후 측정)와 **TRAC**(학습 도중 측정) — 을
MNIST / Fashion-MNIST / CIFAR-10에서 실행하고, 전수 재학습으로 구한 정답 Shapley value(Ground Truth)와
비교해 시각화하는 데스크톱 프로그램입니다.

## 실행

```bash
cd /home/okdol/code/parzen-shapley
/home/okdol/miniconda3/envs/parzen-fl/bin/python -m demo            # 창 실행
/home/okdol/miniconda3/envs/parzen-fl/bin/python -m demo --export artifacts/demo_figs --mode replay   # 창 없이 그림(PNG)만 저장
```

의존성: `parzen-fl` conda 환경(torch, torchvision, Pillow) + `matplotlib`(`pip install matplotlib`).
한글은 Tk 대신 PIL이 그립니다. 폰트는 `demo/fonts/`에 포함된 Pretendard(SIL Open Font License, `LICENSE-Pretendard.txt`)를
쓰고, 없으면 맑은 고딕(`/mnt/c/Windows/Fonts/malgun.ttf`) 등 `demo/config.py`의 `FONT_CANDIDATES` 순서로 찾습니다.
창 제목 표시줄은 Tk가 직접 그려 한글이 깨지므로 영어로 둡니다.

## 화면

1. **메인 화면** — 연합학습과 기여도 평가의 필요성, 두 방법의 개요, 도식. 데이터셋별 오염 참여자 구성
   (`mislabel-k` / `noniid-k`) 드롭다운, 계산 방식(실제 계산 / 저장된 결과 재생), TRAC 라운드 수, 버튼 2개.
2. **로딩 화면** — 데이터셋별 진행률(TRAC은 라운드별 PCC), 로그, 취소.
3. **결과 화면** — (a) 클라이언트별 기여도 막대그래프(오염 참여자는 붉은색), (b) Ground Truth Shapley value 대비
   산점도와 PCC.

## 계산 방식

| 방식 | Parzen-Shapley | TRAC |
|---|---|---|
| 실제 계산 | 데이터셋당 20~40초(첫 실행 시 앵커 모델 학습 포함, GPU). 결과는 `artifacts/demo_cache/`에 캐시 | 라운드당 MNIST/FMNIST ≈ 19초, CIFAR ≈ 8초(RTX 3080). 5라운드 ≈ 4분, 전체 30라운드 ≈ 22분 |
| 저장된 결과 재생 | `results/parzen_shapley.json`의 `prediction_phi` | `results/trac.json` + `results/trac_trajectories/`의 라운드별 궤적(CIFAR-10은 40라운드 run을 30라운드에서 잘라 재생) |

실제 계산한 TRAC은 같은 라운드 수의 저장 궤적과 비트 단위로 일치합니다. Parzen-Shapley는 앵커 모델을 다시
학습하므로 저장값과 ~1e-4(CIFAR ~3e-3) 차이가 납니다.

## 구성

| 파일 | 역할 |
|---|---|
| `demo/app.py` | Tk 앱, 세 화면, 작업 스레드와 이벤트 큐 |
| `demo/backend.py` | `Result`, 설정 목록, 재생/실제 계산, 캐시, `run_job` |
| `demo/plots.py` | 도식·막대·산점도 matplotlib Figure |
| `demo/ktext.py` | PIL 한글 렌더링 위젯(`KLabel`, `KButton`, `KRadiobutton`) |
| `demo/texts.py` | 한국어 문구 |
| `demo/config.py` | 경로, 기본값, 장치 감지 |
| `src/parzen_shapley/` | Parzen-Shapley 방법(앵커·임베딩·Parzen utility·Shapley 추정) |
| `src/trac/` | TRAC 방법(라운드·점수·집계·데이터) |
