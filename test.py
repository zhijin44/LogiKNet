import re

# 重新调整后的标签提取函数
def extract_label(filename):
    # 首先移除文件类型后缀
    label = filename.replace('.pcap.csv', '')
    # 使用正则表达式移除末尾的数字和下划线（假设数字前面可能有下划线）
    label = re.sub(r'[_\d]+(train|test)$', '', label)
    return label

# 示例文件名进行测试
example_filenames = [
    'TCP_IP-DDoS-ICMP4_train.pcap.csv',
    'TCP_IP-DDoS-ICMP1_test.pcap.csv',
    'ARP_Spoofing_test.pcap.csv',
    'Recon-Ping_Sweep_train.pcap.csv'
]

# 测试 extract_label 函数
for filename in example_filenames:
    print(f'{filename} -> {extract_label(filename)}')
