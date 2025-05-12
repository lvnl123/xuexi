import socket
import ffmpeg


def receive_file(host="192.168.127.62", port=12345, save_path="received_audio.pcm"):
    # 创建socket对象
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        # 绑定地址和端口
        server_socket.bind((host, port))
        print(f"正在监听 {host}:{port}...")

        # 开始监听
        server_socket.listen(1)
        print("等待客户端连接...")

        # 接受客户端连接
        client_socket, client_address = server_socket.accept()
        print(f"客户端已连接: {client_address}")
        print("------------------")

        # 接收数据并写入文件
        with open(save_path, 'wb') as f:
            print("正在接收文件...")
            while True:
                data = client_socket.recv(4096)  # 每次接收4096字节
                if not data:
                    break
                f.write(data)
        print(f"文件接收完成，已保存为 {save_path}")

    except Exception as e:
        print(f"发生错误: {e}")

    finally:
        # 关闭连接
        client_socket.close()
        server_socket.close()


def convert_to_mp3(input_path, output_path, sample_rate=16000, channels=1):
    """
    使用 ffmpeg-python 将 PCM 文件转换为 MP3 格式。

    :param input_path: 输入 PCM 文件路径
    :param output_path: 输出 MP3 文件路径
    :param sample_rate: PCM 文件的采样率，默认为 16000 Hz
    :param channels: PCM 文件的声道数，默认为单声道 (1)
    """
    try:
        (
            ffmpeg
            .input(
                input_path,
                format='s16le',  # PCM 格式，小端模式
                acodec='pcm_s16le',
                ar=sample_rate,  # 采样率
                ac=channels  # 声道数
            )
            .output(output_path, acodec='libmp3lame', audio_bitrate='128k')
            .overwrite_output()
            .run()
        )
        print(f"文件已成功转换为 MP3 格式，保存为 {output_path}")
    except ffmpeg.Error as e:
        print(f"FFmpeg 转换失败: {e}")


if __name__ == "__main__":
    # 设置监听地址、端口和保存文件路径
    HOST = "192.168.127.62"  # 监听所有网络接口
    PORT = 12345  # 监听端口
    SAVE_PATH_PCM = "received_audio.pcm"  # 接收到的 PCM 文件名
    SAVE_PATH_MP3 = "output_audio.mp3"  # 转换后的 MP3 文件名

    # 启动接收文件功能
    receive_file(host=HOST, port=PORT, save_path=SAVE_PATH_PCM)

    # 将接收到的 PCM 文件转换为 MP3 格式
    convert_to_mp3(SAVE_PATH_PCM, SAVE_PATH_MP3)