from pathlib import Path
# ----- 입력 -----
METADATA_CSV = r"D:\Stethoscope_Project\Development\OPERA_Copied_from_Ubuntu\data\audio\Segments_from_JSON\metadata.csv"
LINUX_BASE   = "//home//un_wang//my_stethoscope_project//data//audio//Segments_from_JSON"
WIN_BASE     = r"D:\Stethoscope_Project\Development\OPERA_Copied_from_Ubuntu\data\audio\Segments_from_JSON"

# ----- 경로 -----
ROOT      = Path(r"D:\Stethoscope_Project\Deployment\A_OperaStyle")
CACHE_MEL = ROOT/"cache/mels"
CKPTS     = ROOT/"ckpts"
FEATS     = ROOT/"feats"
ARTI      = ROOT/"artifacts"

# ----- OPERA 멜스펙 규격 -----
SR=16000; N_MELS=64; WIN_MS=64; HOP_MS=32
N_FFT=int(SR*(WIN_MS/1000)); HOP=int(SR*(HOP_MS/1000))  # 1024 / 512

# ----- 학습/평가 -----
SEED=42; TEST_SIZE=0.2; VAL_SIZE=0.1
BATCH=32; EPOCHS=30; LR=1e-3
FEAT_DIM=256   # 인코더 출력 임베딩 차원(프레임 단위)
AGG="mean"     # 프레임→세그먼트 집계 방식
