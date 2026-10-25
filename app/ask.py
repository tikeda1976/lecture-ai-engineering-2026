"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

ツールも会話履歴も持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

import httpx
from google import genai
from google.genai import errors, types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"
HTTP_TIMEOUT_SECONDS = 60
HTTP_MAX_ATTEMPTS = 3


def ask(question: str) -> types.GenerateContentResponse:
    """回答と使用量情報を含むレスポンスを返す。表示は呼び出し側で行う。"""
    with genai.Client(
        api_key=get_api_key(),
        http_options=types.HttpOptions(
            # ミリ秒指定。HTTP通信の待機とX-Server-Timeoutヘッダーに使われる。
            # 再試行を含む処理全体の制限時間ではない。
            timeout=HTTP_TIMEOUT_SECONDS * 1000,
            retry_options=types.HttpRetryOptions(attempts=HTTP_MAX_ATTEMPTS),
        ),
    ) as client:
        return client.models.generate_content(model=MODEL, contents=question)


def output_token_count(usage: types.GenerateContentResponseUsageMetadata) -> int | None:
    """回答＋思考のトークン数。未報告の思考分を無条件に0にはしない。"""
    candidates = usage.candidates_token_count
    thoughts = usage.thoughts_token_count
    if candidates is None:
        return None
    if thoughts is not None:
        return candidates + thoughts

    # 入力（キャッシュ分を含む）・回答・ツール分で合計を説明できれば思考分は0。
    tool_tokens = usage.tool_use_prompt_token_count or 0
    if (
        usage.prompt_token_count is not None
        and usage.total_token_count == usage.prompt_token_count + candidates + tool_tokens
    ):
        return candidates
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    print(f"[質問] {args.question}\n")
    try:
        print(
            f"[待機中] APIの応答を待っています（通信・サーバー側タイムアウト設定: {HTTP_TIMEOUT_SECONDS}秒、"
            f"一時的なエラーは初回を含め最大{HTTP_MAX_ATTEMPTS}回試行）。"
            f"総待ち時間は{HTTP_TIMEOUT_SECONDS}秒を超える場合があります。Ctrl+Cで中断できます。",
            flush=True,
        )
        response = ask(args.question)
    except KeyboardInterrupt:
        print("\n[中断] ユーザー操作により終了しました。")
        raise SystemExit(130) from None
    except httpx.TimeoutException:
        print("[タイムアウト] 通信の待機時間を超えました。時間をおいて再実行してください。")
        raise SystemExit(1) from None
    except httpx.HTTPError:
        print("[通信エラー] 通信または応答の読み取りに失敗しました。ネットワーク接続やプロキシ設定を確認してください。")
        raise SystemExit(1) from None
    except errors.APIError as e:
        if e.code == 504:
            print("[タイムアウト] サーバー側で処理が時間切れになりました（504）。質問を短くするか、時間をおいて再実行してください。")
        else:
            print(api_error_message(e.code, e.message))
        raise SystemExit(1) from None

    print(response.text or "テキスト回答を取得できませんでした。安全性によるブロックや、候補が空の可能性があります。")

    usage = response.usage_metadata
    if usage is None:
        print("[トークン数] 使用量情報を取得できませんでした。")
        return

    token_counts = [
        ("入力トークン数", usage.prompt_token_count),
        ("出力トークン数（回答部分）", usage.candidates_token_count),
        ("思考トークン数", usage.thoughts_token_count),
        ("出力トークン数（回答＋思考）", output_token_count(usage)),
        ("合計トークン数（API報告値）", usage.total_token_count),
    ]
    for label, count in token_counts:
        print(f"[{label}] {count if count is not None else '不明'}")


if __name__ == "__main__":
    main()
