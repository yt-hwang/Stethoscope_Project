@echo off
cd /d D:\Stethoscope_Project\Deployment\B_Pretrained
python 01_index_from_metadata.py || goto :eof
python 02_cache_mels.py         || goto :eof
python 03_embed_efficientnet.py || goto :eof
python 04_linear_probe.py       || goto :eof
echo [DONE]
pause
