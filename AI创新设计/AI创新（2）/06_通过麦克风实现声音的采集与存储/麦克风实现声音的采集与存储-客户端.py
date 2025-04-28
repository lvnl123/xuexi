import subprocess

# 测试 FFmpeg 是否可用
try:
    result = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print("FFmpeg 已找到！版本信息如下：")
    print(result.stdout)
except FileNotFoundError:
    print("未找到 FFmpeg，请确保已安装并添加到系统路径中。")