from flask import Flask, request, jsonify
from dotenv import load_dotenv
from flask_cors import CORS
import traceback
import openai
import os

# OpenAIクライアントの初期化（APIキーは安全に管理してください）
load_dotenv()

# 環境変数からAPIキー取得
openai_api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
@app.route("/")
def index():
    return f"OpenAIキーの一部: {openai_api_key[:5]}******"
# -----------------------
# ① /childdata: イベント生成（3行構成）
# -----------------------
@app.route("/childdata", methods=["POST"])
def handle_child_data():
    try:
        data = request.get_json()
        print("✅ Unityから受信 (childdata):", data)

        prompt = build_event_prompt(data)
        result = call_gpt(prompt, system_prompt_event())
        return result, 200, {
            'Content-Type': 'text/plain; charset=utf-8',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': '*',
            'Access-Control-Allow-Methods': 'POST, OPTIONS'
        }


    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# -----------------------
# ② /feedback: 声かけによる成長（6行構成）
# -----------------------
@app.route("/feedback", methods=["POST"])
def handle_feedback():
    try:
        data = request.get_json()
        print("✅ Unityから受信 (feedback):", data)

        prompt = build_feedback_prompt(data)
        result = call_gpt(prompt, system_prompt_feedback())
        return result, 200, {'Content-Type': 'text/plain; charset=utf-8'}

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# -----------------------
# ③ /ending: 20歳以降の人生（物語出力）
# -----------------------
@app.route("/ending", methods=["POST"])
def handle_ending():
    try:
        data = request.get_json()
        print("✅ Unityから受信 (ending):", data)

        prompt = build_ending_prompt(data)
        result = call_gpt(prompt, system_prompt_ending())
        return result, 200, {'Content-Type': 'text/plain; charset=utf-8'}

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# -----------------------
# 共通：GPT呼び出し関数
# -----------------------
def call_gpt(user_prompt, system_prompt):
    result = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7
    )
    return result.choices[0].message.content.strip()

# -----------------------
# SYSTEM PROMPTS（出力形式指定）
# -----------------------

def system_prompt_event():
    return (
        "以下の情報を元に、3行の育成イベントを出力してください。年齢に沿った出力を行ってください。\n"
        "【出力形式】\n"
        "1行目: イベントタイトル\n"
        "2行目: イベント内容（その子に起きたできごと）\n"
        "3行目: 子供のセリフ\n\n"
        "【例】\n"
        "誕生日のお祝い\n"
        "今日は3歳の誕生日。家族とケーキを囲んだ。\n"
        "「ふーってしたよ！」"
    )

def system_prompt_feedback():
    return (
        "以下の子供の情報と出来事をもとに、プレイヤーの声かけによる影響を分析しなさい。\n"
        "必ず出力は7行、次の形式に従ってください（ラベルなし、改行で区切る）：\n"
        "1行目: 性格パラメータ5つ（float）\n"
        "2行目: 能力パラメータ5つ（float）\n"
        "3行目: パラメータの変化理由（どのパラメータがどういう理由で変化したかを記述）\n"
        "4行目: スキル（パラメータやイベントに応じて獲得しそうなスキルを記載、例：歌がうまい、サッカーが得意）\n"
        "5行目: スキルが獲得できる確率（float, 0.0〜1.0, パラメータやイベントに応じて評価）\n"
        "6行目: 夢の実現スコア（float, 0.0〜1.0, 夢に対して実現の可能性を評価）\n"
        "7行目: 愛ゲージスコア（float, 0.0〜1.0, 得ている愛情を評価）\n"
        "【出力例】\n"
        "2.1 3.4 1.8 4.0 2.7\n"
        "1.0 4.2 3.3 2.9 3.1\n"
        "「すごいね！」という声かけで自信が高まりました。\n"
        "絵が上手\n"
        "0.3\n"
        "0.4\n"
        "0.7"
    )


def system_prompt_ending():
    return (
        "以下の育成情報をもとに、その子の20歳以降の人生を感動的な物語として1段落で出力してください。"
    )

# -----------------------
# USER PROMPT BUILDERS
# -----------------------

def build_event_prompt(data):
    return f"""
名前: {data.get("name", "")}
年齢: {data.get("age", 0)}
夢: {data.get("dream", "")}

性格パラメータ（対応順: 創造性, 外向性, 協調性, 勤勉性, 情動性）: {data.get("p", [])}
能力パラメータ（対応順: 認知能力, 運動能力, 好奇心, 自己肯定感, 外見）: {data.get("a", [])}
スキル: {data.get("skills", [])}
"""


def build_feedback_prompt(data):
    return f"""
名前: {data.get("name", "")}
年齢: {data.get("age", 0)}
夢: {data.get("dream", "")}

性格パラメータ（対応する順に: 創造性, 外向性, 協調性, 勤勉性, 情動性）: {data.get("p", [])}
能力パラメータ（対応する順に: 認知能力, 運動能力, 好奇心, 自己肯定感, 外見）: {data.get("a", [])}
スキル: {data.get("skills", [])}

イベントタイトル: {data.get("eventTitle", "")}
イベント内容: {data.get("eventContent", "")}
子供の発言: {data.get("childUtterance", "")}

親の声かけ: 「{data.get("parentComment", "")}」
"""


def build_ending_prompt(data):
    return f"""
名前: {data.get("name", "")}
夢: {data.get("dream", "")}
年齢: {data.get("age", 20)}

性格パラメータ（対応順: 創造性, 外向性, 協調性, 勤勉性, 情動性）: {data.get("p", [])}
能力パラメータ（対応順: 認知能力, 運動能力, 好奇心, 自己肯定感, 外見）: {data.get("a", [])}
スキル: {data.get("skills", [])}
愛ゲージ: {data.get("loveGauge", 0.0)}
夢の実現スコア: {data.get("dreamRealization", 0.0)}
"""


# -----------------------
# サーバ起動
# -----------------------
if __name__ == "__main__":
    app.run(debug=True)


