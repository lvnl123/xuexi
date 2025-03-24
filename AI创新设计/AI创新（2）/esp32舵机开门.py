import socket

def start_server(host='192.168.110.62', port=8080):
    """
    启动 Python 服务端，等待客户端连接并发送指令
    """
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    print(f"Server listening on {host}:{port}")

    client_socket, client_address = server_socket.accept()
    print(f"Client connected from {client_address}")

    while True:
        try:
            # 接收用户输入
            command = input("Enter command (OPEN/CLOSE): ").strip().upper()
            if command in ['1', '0']:
                client_socket.sendall(command.encode('utf-8'))  # 发送指令到微控制器
                print(f"Sent command: {command}")
            else:
                print("Invalid command. Please enter OPEN or CLOSE.")
        except Exception as e:
            print("Error:", e)
            break

    client_socket.close()
    server_socket.close()

if __name__ == "__main__":
    start_server()