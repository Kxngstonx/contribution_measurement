"""Korean UI strings for the demo (kept in one place)."""

WINDOW_TITLE = "Federated Learning Contribution Demo: Parzen-Shapley / TRAC"
TITLE = "연합학습 참여자 기여도 평가 데모: Parzen-Shapley · TRAC"

INTRO_FL = (
    "연합학습(Federated Learning)은 여러 참여자(클라이언트)가 자신의 데이터를 외부로 "
    "내보내지 않고 공동의 글로벌 모델을 학습하는 방식입니다. 각 참여자는 서버에서 받은 "
    "글로벌 모델을 자기 데이터로 학습한 뒤 모델의 변화량(업데이트)만 서버에 보내고, "
    "서버는 이를 모아 글로벌 모델을 갱신합니다."
)

INTRO_WHY = (
    "참여자마다 데이터의 품질이 다릅니다. 라벨이 잘못 붙은 데이터나 일부 클래스에 "
    "치우친 데이터를 가진 참여자(오염 참여자)는 글로벌 모델의 성능을 떨어뜨립니다. "
    "모든 참여자에게 같은 보상을 주면 공정하지 않으므로, 각 참여자가 글로벌 모델의 "
    "성능에 얼마나 기여했는지(기여도)를 측정해야 합니다. 기준이 되는 기여도는 "
    "Shapley value로, 참여자 조합마다 모델을 다시 학습해야 해 계산 비용이 매우 큽니다. "
    "이 프로그램은 재학습 없이 기여도를 측정하는 두 방법을 보여주고, 전수 재학습으로 "
    "구한 정답 Shapley value(Ground Truth)와 비교합니다."
)

METHOD_PARZEN_TITLE = "방법 1 — Parzen-Shapley (학습이 끝난 뒤 측정)"
METHOD_PARZEN = (
    "참여자 데이터 전체로 학습한 모델 하나를 고정(anchor)하고, 그 모델의 임베딩 위에서 "
    "클래스별 Parzen 밀도로 참여자 조합의 성능(utility)을 추정합니다. 조합마다 모델을 "
    "재학습하지 않고, balanced permutation Monte Carlo로 방문한 조합만 평가해 각 "
    "참여자의 Shapley value를 구합니다. 설정당 수 초면 끝납니다."
)

METHOD_TRAC_TITLE = "방법 2 — TRAC (학습 도중 측정)"
METHOD_TRAC = (
    "서버가 소량의 서버 데이터(128개)로 매 라운드 한 걸음 학습한 서버 업데이트를 측정 "
    "기준으로 삼고, 각 참여자 업데이트와의 정렬 정도(코사인 유사도)를 라운드마다 누적(EMA)"
    "해 기여도로 사용합니다. 서버 데이터의 클래스 분포가 치우쳐 있어도 손실함수를 목표 "
    "클래스 분포에 맞게 보정하고, 누적 기여도를 집계 가중치로 써서 오염 참여자의 영향을 "
    "줄입니다. 참여자 쪽에는 추가 연산이 없습니다."
)

CONTROLS_TITLE = "평가 설정"
SETTING_HINT = "데이터셋별 오염 참여자 구성 (mislabel-k: 라벨 오류 k명, noniid-k: 클래스 편향 k명)"
MODE_LABEL = "계산 방식"
MODE_REAL = "실제 계산 (한 번 계산한 결과는 캐시)"
MODE_REPLAY = "저장된 결과 재생"
ROUNDS_LABEL = "TRAC 라운드 수"
ROUNDS_HINT = "{rounds}라운드 ≈ {minutes:.1f}분 (전체 30라운드 ≈ {full:.0f}분)"
DEVICE_CPU_WARNING = "경고: GPU가 감지되지 않아 CPU로 계산합니다. 실제 계산이 매우 느립니다."

BTN_PARZEN = "Parzen-Shapley 기여도 평가"
BTN_TRAC = "TRAC 기여도 평가"
BTN_BACK = "메인으로"
BTN_CANCEL = "취소"

LOADING_TITLE = {"parzen": "Parzen-Shapley 기여도 평가 중", "trac": "TRAC 기여도 평가 중"}
LOADING_MODE = {"real": "실제 계산", "replay": "저장된 결과 재생"}
STATUS_WAITING = "대기 중"
STATUS_CANCELLING = "취소 중… (진행 중인 라운드가 끝나면 멈춥니다)"
STATUS_DONE = "완료  PCC {pcc:.3f}"
STATUS_ROUND = "라운드 {round}/{rounds}  PCC {pcc:.3f}"
STATUS_STAGE = "{stage} ({step}/{total})"

PARZEN_STAGES = ("앵커 모델 임베딩 추출", "Parzen utility 구성", "balanced MC Shapley 추정")
LOG_ANCHOR_TRAIN = "앵커 모델 학습 중 (5 epoch, GPU 약 5초)"
LOG_ANCHOR_FOUND = "저장된 앵커 모델 사용"
LOG_CACHE_HIT = "캐시된 결과를 불러왔습니다"
LOG_REPLAY = "저장된 결과를 재생합니다"
LOG_DATASET_START = "{title} {label}: 평가 시작"
LOG_DATASET_DONE = "{title} {label}: 완료  PCC {pcc:.3f}  ({elapsed:.1f}초)"

RESULT_TITLE = {"parzen": "Parzen-Shapley 기여도 평가 결과", "trac": "TRAC 기여도 평가 결과"}
RESULT_SOURCE = {"real": "실제 계산", "replay": "저장된 결과", "cache": "캐시"}
RESULT_LINE = "{title} {label}: PCC {pcc:.3f} ({source}{extra})"
RESULT_ACCURACY = ", 글로벌 모델 정확도 {accuracy:.3f}"
RESULT_ROUNDS = ", {rounds}라운드"

FIG_BAR_TITLE = "{method} — 클라이언트별 기여도"
FIG_SCATTER_TITLE = "{method} — 정답(Ground Truth) Shapley value와의 비교"
FIG_YLABEL = "기여도 점수"
FIG_XLABEL_CLIENT = "클라이언트"
FIG_XLABEL_GT = "Ground Truth Shapley value"
FIG_CLEAN = "정상 참여자"
FIG_CONTAM = "오염 참여자"
FIG_PCC = "PCC = {pcc:.3f}"
METHOD_NAMES = {"parzen": "Parzen-Shapley", "trac": "TRAC"}

SCHEMATIC_CLIENT = "참여자"
SCHEMATIC_SERVER = "서버\n(글로벌 모델)"
SCHEMATIC_UPDATE = "업데이트"
SCHEMATIC_SCORE = "기여도"

ERROR_TITLE = "오류"
ERROR_BODY = "기여도 평가 중 오류가 발생했습니다.\n\n{text}"
