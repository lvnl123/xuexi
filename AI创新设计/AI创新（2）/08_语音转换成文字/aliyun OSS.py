import oss2

# 阿里云OSS的相关信息
access_key_id = 'LTAI5tBtM4nJaWuQy8krAD88'  # 替换为你的AccessKeyId
access_key_secret = 'SZ7XGPfuRzqnKnMEgpOWhF9LaweAOS'  # 替换为你的AccessKeySecret
bucket_name = 'esp32yuyin'  # 替换为你的Bucket名称
endpoint = 'http://oss-cn-hangzhou.aliyuncs.com'  # 替换为你的OSS Endpoint，例如：'http://oss-cn-hangzhou.aliyuncs.com'

# 创建Bucket对象
auth = oss2.Auth(access_key_id, access_key_secret)
bucket = oss2.Bucket(auth, endpoint, bucket_name)

# 上传文件到OSS
def upload_file_to_oss(file_path, object_name):
    try:
        # 上传文件
        with open(file_path, 'rb') as fileobj:
            bucket.put_object(object_name, fileobj)
        print(f"文件上传成功: {object_name}")
    except Exception as e:
        print(f"文件上传失败: {e}")

if __name__ == "__main__":
    # 本地音频文件路径
    local_audio_path = 'E:\learn\pythonProject\OpenCV语音识别\output_audio.mp3'  # 替换为你要上传的音频文件路径
    # OSS上的文件名
    oss_audio_name = 'example.mp3'  # 替换为你想在OSS上保存的音频文件名

    # 调用上传函数
    upload_file_to_oss(local_audio_path, oss_audio_name)