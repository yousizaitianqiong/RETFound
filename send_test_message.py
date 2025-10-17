import pika
import json
import random

# ===== RabbitMQ 配置 =====
RABBITMQ_HOST = "192.168.10.215"
RABBITMQ_PORT = 5672
RABBITMQ_USER = "guest"
RABBITMQ_PASS = "guest"

# ===== 队列配置 =====
QUEUES = {
    "check_queue": {
    "exchange": "check_exchange",
    "routing_key": "check_routing_key",
    "data": {
        "osAnteriorImg": "/admin/sys-file/aipluseyes/afe15e7d10db40c2abc0803c854c12f3.png",
        "odAnteriorImg": "/admin/sys-file/aipluseyes/afe15e7d10db40c2abc0803c854c12f3.png",
        "osFundusImg": "/admin/sys-file/aipluseyes/afe15e7d10db40c2abc0803c854c12f3.png",
        "odFundusImg": "/admin/sys-file/aipluseyes/afe15e7d10db40c2abc0803c854c12f3.png",
        "id": "1967856474378854401"
    }
},

    "report_queue": {
        "exchange": "report_exchange",
        "routing_key": "report_routing_key",
        "data_list": [
            {
                "anteriorReport": "双眼前节正常，角膜透明，前房深度正常，虹膜及晶状体无异常。",
                "fundusReport": "双眼眼底结构正常，视网膜、脉络膜及视神经盘清晰，血管走行规则。",
            },
            {
                "anteriorReport": "右眼角膜轻度浸润，前房无明显积液，晶状体轻微混浊。",
                "fundusReport": "左眼眼底轻微微血管曲张，视网膜未见出血或渗出。",
            },
            {
                "anteriorReport": "双眼前节可见中度炎症，角膜散在小面积溃疡。",
                "fundusReport": "右眼视网膜出血点明显，视神经盘水肿轻度，左眼眼底大体正常。",
            },
            {
                "anteriorReport": "双眼前节明显炎症，角膜大面积溃疡，前房积脓。",
                "fundusReport": "双眼眼底大片出血，视神经盘水肿明显，视网膜脱离风险高。",
            },
            {
                "anteriorReport": "双眼前节未见明显病变，但角膜轻度散在点状浸润。",
                "fundusReport": "双眼眼底血管走行稍不规则，视网膜无出血。",
            },
        ]
    }
}

def send_message(queue_name: str):
    cfg = QUEUES.get(queue_name)
    if not cfg:
        print(f"❌ 未知队列: {queue_name}")
        return

    # 建立连接
    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host=RABBITMQ_HOST, port=RABBITMQ_PORT, credentials=credentials)
    )
    channel = connection.channel()

    # 声明交换机
    channel.exchange_declare(exchange=cfg["exchange"], exchange_type='direct', durable=True)

    # 发送消息
    if queue_name == "report_queue":
        # 从 data_list 随机选一条
        message_data = random.choice(cfg["data_list"])
        # 添加唯一 id
        message_data["id"] = str(random.randint(1000000000000000000, 9999999999999999999))
    else:
        message_data = cfg["data"]

    body = json.dumps(message_data, ensure_ascii=False)
    channel.basic_publish(
        exchange=cfg["exchange"],
        routing_key=cfg["routing_key"],
        body=body.encode("utf-8"),
        properties=pika.BasicProperties(delivery_mode=2)
    )

    print(f"✅ 已发送到 {queue_name}: {body}")
    connection.close()

if __name__ == "__main__":
    print("📦 请选择队列发送：")
    print("1️⃣ check_queue (审核任务)")
    print("2️⃣ report_queue (报告生成)")
    choice = input("输入 1/2，默认 1：").strip() or "1"
    send_message("check_queue" if choice == "1" else "report_queue")
