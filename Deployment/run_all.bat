@echo off
cd /d D:\Stethoscope_Project\Deployment\A_OperaStyle
python 01_index_from_metadata.py || goto :eof
python 02_cache_mels.py         || goto :eof
python 03_ssl_contrastive_train.py || goto :eof
python 04_extract_features_from_ckpt.py || goto :eof
python 05_linear_probe.py       || goto :eof
echo [DONE]
pause
