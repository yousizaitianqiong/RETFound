import pika
import json
import traceback
from ai_diagnosis import load_retfound_model, run_ai_diagnosis,run_ai_diagnosis_debug

BASE_URL = "http://192.168.10.215:9999/"



# ================= RabbitMQ 配置 =================
RABBITMQ_HOST = "192.168.10.215"
RABBITMQ_PORT = 5672
RABBITMQ_USER = "guest"
RABBITMQ_PASS = "guest"

# 队列/交换机/路由键
CHECK_QUEUE = "check_queue"
CHECK_EXCHANGE = "check_exchange"
CHECK_ROUTING_KEY = "check_routing_key"

RESULT_CHECK_QUEUE = "result_check_queue"
RESULT_CHECK_EXCHANGE = "result_check_exchange"
RESULT_CHECK_ROUTING_KEY = "result_check_routing_key"

# ================= 占位前节结果 =================
def placeholder_anterior_result():
    return {
        "conjunctivitis": 0,
        "keratitis": 0,
        "cataract": 0,
        "pterygium": 0,
        "exfoliationSyndrome": 0
    }

# ================= 眼底模型加载 =================
MODEL_PATH = "output/best_model.pth"  # TODO: 修改为你的模型路径

fundus_model = load_retfound_model(
    model_name="RETFound_mae",
    model_path=MODEL_PATH,  # ⚠️ 必须传
)



import time
import requests
from PIL import Image
from io import BytesIO

def run_fundus_model(img_path, max_retries=3, retry_delay=1):
    """调用 RETFound 模型进行眼底诊断，仅通过 URL 下载图片，支持重试"""
    img_url = BASE_URL + img_path
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(img_url, stream=True, timeout=5)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")
            break  # 下载成功就跳出循环
        except requests.exceptions.RequestException as e:
            print(f"❌ 第 {attempt} 次下载失败: {img_url}, 错误: {e}")
            if attempt < max_retries:
                print(f"⏳ {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
            else:
                print("❌ 已达到最大重试次数，下载失败，返回 None")
                return None

    # 调用模型
    result = run_ai_diagnosis_debug(img, fundus_model)

    # 调试打印 top5 概率
    if result:
        print("🔹 模型诊断结果:")
        print(f"  结论: {result['conclusion']}")
        print(f"  置信度: {result['confidence']:.4f}")
        print(f"  top5 classes: {result['details']['top5_classes']}")
        print(f"  top5 probs: {result['details']['top5_probs']}")

    # 转换为你需要的结构
    mapping = {
        "AMD-CFP": "AMD",
        "CSC-CFP": "CSC",
        "RP-CFP": "RP",
        "ON": "ON",
        "GLA": "GLA",
        "RVO-CFP": "RVO",
        "normal-CFP": "normal",
        "其他": "other"
    }

    # 初始化输出
    output = {"AMD":0,"CSC":0,"RP":0,"ON":0,"GLA":0,"DR":0,"RVO":0}
    if result and result["class_name"] in mapping:
        key = mapping[result["class_name"]]
        output[key] = 1

    return output

# ================= 处理消息 =================
def handle_check_message(ch, method, properties, body):
    try:
        message = json.loads(body)
        print(f"\n🧾 收到审核任务: {json.dumps(message, indent=2, ensure_ascii=False)}")

        osAnteriorImg = message.get("osAnteriorImg")
        odAnteriorImg = message.get("odAnteriorImg")
        osFundusImg = message.get("osFundusImg")
        odFundusImg = message.get("odFundusImg")
        report_id = message.get("id")

        # ================= 模型推理 =================
        osAnteriorResult = placeholder_anterior_result()
        odAnteriorResult = placeholder_anterior_result()
        osFundusResult = run_fundus_model(osFundusImg)
        odFundusResult = run_fundus_model(odFundusImg)

        # ================= 生成结果 =================
        result = {
            "osAnteriorResult": osAnteriorResult,
            "odAnteriorResult": odAnteriorResult,
            "osFundusResult": osFundusResult,
            "odFundusResult": odFundusResult,
            "id": report_id
        }

        # ================= 发送到结果队列 =================
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=RABBITMQ_HOST,
                port=RABBITMQ_PORT,
                credentials=pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
            )
        )
        channel = connection.channel()
        channel.exchange_declare(exchange=RESULT_CHECK_EXCHANGE, exchange_type='direct', durable=True)
        channel.basic_publish(
            exchange=RESULT_CHECK_EXCHANGE,
            routing_key=RESULT_CHECK_ROUTING_KEY,
            body=json.dumps(result, ensure_ascii=False).encode("utf-8"),
            properties=pika.BasicProperties(delivery_mode=2)
        )
        connection.close()

        print(f"✅ 已发送审核结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as e:
        print("❌ 处理消息时出错:", e)
        traceback.print_exc()
        ch.basic_nack(delivery_tag=method.delivery_tag)

# ================= 主监听函数 =================
def main():
    print(f"🚀 审核服务已启动，正在连接 RabbitMQ {RABBITMQ_HOST}:{RABBITMQ_PORT} ...")
    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host=RABBITMQ_HOST, port=RABBITMQ_PORT, credentials=credentials)
    )
    channel = connection.channel()

    # 声明队列和交换机
    channel.exchange_declare(exchange=CHECK_EXCHANGE, exchange_type='direct', durable=True)
    channel.queue_declare(
        queue=CHECK_QUEUE,
        durable=True,
        arguments={
            'x-dead-letter-exchange': 'dlx_check_exchange',
            'x-dead-letter-routing-key': 'dlx_check_routing_key'
        }
    )


    channel.queue_bind(exchange=CHECK_EXCHANGE, queue=CHECK_QUEUE, routing_key=CHECK_ROUTING_KEY)

    # 监听队列
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=CHECK_QUEUE, on_message_callback=handle_check_message, auto_ack=False)

    print(f"📡 正在监听队列: {CHECK_QUEUE}")
    channel.start_consuming()

if __name__ == "__main__":
    main()
