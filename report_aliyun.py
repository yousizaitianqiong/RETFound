import pika
import json
import dashscope
from dashscope import Generation
import traceback

# =====================================
# 🧩 基本配置
# =====================================
RABBITMQ_HOST = '192.168.10.215'
RABBITMQ_PORT = 5672
RABBITMQ_USER = 'guest'
RABBITMQ_PASS = 'guest'

DASH_SCOPE_API_KEY = 'sk-649a68f2af5248348307e70b7e57a44f'

# 输入队列
REPORT_QUEUE = 'report_queue'
REPORT_EXCHANGE = 'report_exchange'
REPORT_ROUTING_KEY = 'report_routing_key'

# 输出队列
RESULT_REPORT_QUEUE = 'result_report_queue'
RESULT_REPORT_EXCHANGE = 'result_report_exchange'
RESULT_REPORT_ROUTING_KEY = 'result_report_routing_key'

# 初始化 DashScope
dashscope.api_key = DASH_SCOPE_API_KEY


# =====================================
# 🧠 调用阿里云语言模型生成报告内容
# =====================================
def generate_report(anterior_report, fundus_report):
    prompt = f"""
你是一名资深眼科医生，请根据以下检查结果生成报告解读。

前节检查结果：{anterior_report}
眼底检查结果：{fundus_report}

请输出四个字段，格式严格如下（使用 JSON）：
{{
  "anteriorInterpret": "...",
  "anteriorAdvice": "...",
  "fundusInterpret": "...",
  "fundusAdvice": "..."
}}
要求：内容专业、简洁明了、避免重复表达。
    """

    try:
        response = Generation.call(
            model="qwen-plus",        # 可换为 "qwen-turbo" 或 "qwen2-7b-instruct"
            prompt=prompt,
            result_format="text"      # 返回纯文本
        )

        # ✅ 新版 SDK：response 是一个 dict-like 对象
        text = ""
        if isinstance(response, dict):
            text = response.get("output", {}).get("text", "")
        else:
            # 兼容旧 SDK
            text = getattr(response, "output_text", "")

        if not text:
            print("❌ AI返回为空或格式不符：", response)
            return ("", "", "", "")

        print("🧠 AI返回原文：", text)

        # 解析 JSON 格式的输出
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # 若AI未返回标准JSON，则尝试提取大括号内容
            start = text.find("{")
            end = text.rfind("}") + 1
            json_str = text[start:end]
            data = json.loads(json_str) if json_str else {}

        return (
            data.get("anteriorInterpret", ""),
            data.get("anteriorAdvice", ""),
            data.get("fundusInterpret", ""),
            data.get("fundusAdvice", "")
        )

    except Exception as e:
        print("❌ AI生成失败：", e)
        traceback.print_exc()
        return ("", "", "", "")


def extract_between(text, start_marker, end_marker):
    """简单提取两个标记之间的内容"""
    try:
        start = text.index(start_marker) + len(start_marker)
        end = text.index(end_marker)
        return text[start:end].strip()
    except ValueError:
        return ""


# =====================================
# 📨 处理报告生成消息
# =====================================
def handle_report_message(ch, method, properties, body):
    try:
        message = json.loads(body)
        print("\n🧾 收到报告生成任务：", json.dumps(message, ensure_ascii=False, indent=2))

        anterior_report = message.get("anteriorReport", "")
        fundus_report = message.get("fundusReport", "")
        report_id = message.get("id")

        # 调用阿里云生成报告解读
        anterior_interpret, anterior_advice, fundus_interpret, fundus_advice = generate_report(anterior_report, fundus_report)

        result = {
            "anteriorAdvice": anterior_advice,
            "anteriorInterpret": anterior_interpret,
            "fundusAdvice": fundus_advice,
            "fundusInterpret": fundus_interpret,
            "id": report_id
        }

        # 发送结果到 result_report_queue
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=RABBITMQ_HOST,
                port=RABBITMQ_PORT,
                credentials=pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
            )
        )
        channel = connection.channel()
        channel.exchange_declare(exchange=RESULT_REPORT_EXCHANGE, exchange_type='direct', durable=True)
        channel.basic_publish(
            exchange=RESULT_REPORT_EXCHANGE,
            routing_key=RESULT_REPORT_ROUTING_KEY,
            body=json.dumps(result, ensure_ascii=False)
        )

        print("✅ 已发送报告解读结果：", json.dumps(result, ensure_ascii=False, indent=2))
        connection.close()
        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as e:
        print("❌ 处理消息时出错：", e)
        traceback.print_exc()
        ch.basic_nack(delivery_tag=method.delivery_tag)


# =====================================
# 🚀 主监听函数
# =====================================
def main():
    print("🚀 报告生成解读服务已启动，正在连接 RabbitMQ...")

    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host=RABBITMQ_HOST, port=RABBITMQ_PORT, credentials=credentials)
    )
    channel = connection.channel()

    # 声明队列绑定
    channel.exchange_declare(exchange=REPORT_EXCHANGE, exchange_type='direct', durable=True)

    # ✅ 必须与已有RabbitMQ队列配置完全一致！
    args = {
        'x-dead-letter-exchange': 'dlx_report_exchange',
        'x-dead-letter-routing-key': 'dlx_report_routing_key'
    }
    channel.queue_declare(queue=REPORT_QUEUE, durable=True, arguments=args)

    channel.queue_bind(exchange=REPORT_EXCHANGE, queue=REPORT_QUEUE, routing_key=REPORT_ROUTING_KEY)

    # 监听队列
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=REPORT_QUEUE, on_message_callback=handle_report_message, auto_ack=False)

    print(f"📡 正在监听队列: {REPORT_QUEUE} (RabbitMQ {RABBITMQ_HOST}:5672)")
    channel.start_consuming()


if __name__ == "__main__":
    main()
