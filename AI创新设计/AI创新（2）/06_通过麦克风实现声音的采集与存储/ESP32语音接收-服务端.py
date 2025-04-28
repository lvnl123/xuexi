import socket
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


if __name__ == "__main__":
    # 设置监听地址、端口和保存文件路径
    HOST = "192.168.127.62"  # 监听所有网络接口
    PORT = 12345  # 监听端口
    SAVE_PATH = "received_audio.pcm"  # 保存的文件名

    # 启动接收文件功能
    receive_file(host=HOST, port=PORT, save_path=SAVE_PATH)
