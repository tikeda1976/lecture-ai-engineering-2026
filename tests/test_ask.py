"""第1回の回答・使用量表示を、実APIを呼ばずに確認する。"""

from unittest.mock import MagicMock, Mock

import pytest
import httpx
from google.genai import errors, types

from app import ask


def test_ask_returns_response_without_printing(monkeypatch, capsys):
    response = types.GenerateContentResponse()
    client = MagicMock()
    client.__enter__.return_value = client
    client.models.generate_content.return_value = response
    monkeypatch.setattr(ask.genai, "Client", Mock(return_value=client))
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")

    assert ask.ask("質問") is response
    assert capsys.readouterr().out == ""
    client.models.generate_content.assert_called_once_with(model=ask.MODEL, contents="質問")
    options = ask.genai.Client.call_args.kwargs["http_options"]
    assert options.timeout == 60_000
    assert options.retry_options.attempts == 3
    client.__exit__.assert_called_once()


@pytest.mark.parametrize("thoughts, expected", [(5, "25"), (0, "20"), (None, "20")])
def test_main_displays_answer_then_usage(monkeypatch, capsys, thoughts, expected):
    response = types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=[types.Part(text="回答本文")]))],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=10,
            candidates_token_count=20,
            thoughts_token_count=thoughts,
            total_token_count=35 if thoughts == 5 else 30,
        ),
    )
    monkeypatch.setattr(ask, "ask", lambda question: response)
    monkeypatch.setattr("sys.argv", ["ask", "質問"])
    ask.main()
    output = capsys.readouterr().out
    assert output.index("回答本文") < output.index("[入力トークン数]")
    assert "[出力トークン数（回答部分）] 20" in output
    assert f"[出力トークン数（回答＋思考）] {expected}" in output
    assert f"[合計トークン数（API報告値）] {response.usage_metadata.total_token_count}" in output


def test_main_explains_missing_text_and_usage(monkeypatch, capsys):
    monkeypatch.setattr(ask, "ask", lambda question: types.GenerateContentResponse())
    monkeypatch.setattr("sys.argv", ["ask", "質問"])
    ask.main()
    output = capsys.readouterr().out
    assert "テキスト回答を取得できませんでした" in output
    assert "使用量情報を取得できませんでした" in output
    assert "None" not in output


@pytest.mark.parametrize(
    "error, exit_code, message",
    [(httpx.ReadTimeout("test timeout"), 1, "[タイムアウト]"),
     (httpx.ConnectError("test connection failure"), 1, "[通信エラー]"),
     (httpx.ReadError("test connection reset"), 1, "[通信エラー]"),
     (httpx.DecodingError("invalid encoding"), 1, "[通信エラー]"),
     (httpx.TooManyRedirects("redirect loop"), 1, "[通信エラー]"),
     (errors.APIError(504, {"error": {"message": "deadline"}}), 1, "[タイムアウト]"),
     (errors.APIError(429, {"error": {"message": "quota"}}), 1, "レート制限"),
     (errors.APIError(503, {"error": {"message": "busy"}}), 1, "混み合っています"),
     (errors.APIError(403, {"error": {"message": "denied"}}), 1, "APIキーかモデル名"),
     (KeyboardInterrupt(), 130, "[中断]")],
)
def test_main_handles_errors_and_interrupt(monkeypatch, capsys, error, exit_code, message):
    client = MagicMock()
    client.__enter__.return_value = client
    client.models.generate_content.side_effect = error
    monkeypatch.setattr(ask.genai, "Client", Mock(return_value=client))
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")
    monkeypatch.setattr("sys.argv", ["ask", "質問"])

    with pytest.raises(SystemExit) as exc:
        ask.main()

    assert exc.value.code == exit_code
    output = capsys.readouterr().out
    assert output.index("[待機中]") < output.index(message)
    assert "Traceback" not in output
    assert "[入力トークン数]" not in output
    client.__exit__.assert_called_once()


@pytest.mark.parametrize(
    "counts, expected",
    [
        ({"prompt_token_count": 21, "candidates_token_count": 503, "total_token_count": 524}, 503),
        ({"candidates_token_count": 503}, None),
        ({"prompt_token_count": 21, "candidates_token_count": 503, "total_token_count": 530}, None),
        ({"thoughts_token_count": 5}, None),
        ({"candidates_token_count": 0, "thoughts_token_count": 0}, 0),
        ({"prompt_token_count": 21, "cached_content_token_count": 10,
          "candidates_token_count": 503, "total_token_count": 524}, 503),
        ({"prompt_token_count": 21, "candidates_token_count": 503,
          "total_token_count": 534, "tool_use_prompt_token_count": 10}, 503),
        ({"prompt_token_count": 21, "candidates_token_count": 503,
          "total_token_count": 524, "tool_use_prompt_token_count": 10}, None),
    ],
)
def test_output_token_count_preserves_unknown_values(counts, expected):
    assert ask.output_token_count(types.GenerateContentResponseUsageMetadata(**counts)) == expected


@pytest.mark.parametrize("status", [429, 503, 504])
@pytest.mark.parametrize("recover", [True, False])
def test_real_sdk_retries_http_errors(monkeypatch, status, recover):
    """実SDKを通し、外部通信なしで試行回数とタイムアウトヘッダーを確認する。"""
    requests = []

    def handle(request):
        requests.append(request)
        assert request.headers["X-Server-Timeout"] == str(ask.HTTP_TIMEOUT_SECONDS)
        if recover and len(requests) == 2:
            return httpx.Response(200, json={
                "candidates": [{"content": {"parts": [{"text": "成功"}], "role": "model"}}],
            })
        return httpx.Response(status, json={"error": {"code": status, "message": "temporary"}})

    original_client = ask.genai.Client

    def make_client(*, api_key, http_options):
        # 待機だけを短縮し、ask()の試行回数・SDKの判定ロジックはそのまま使う。
        http_options.retry_options.max_delay = 0.001
        http_options.client_args = {"transport": httpx.MockTransport(handle)}
        return original_client(api_key=api_key, vertexai=False, http_options=http_options)

    monkeypatch.setattr(ask.genai, "Client", make_client)
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")
    if recover:
        assert ask.ask("質問").text == "成功"
        assert len(requests) == 2
    else:
        with pytest.raises(errors.APIError) as exc:
            ask.ask("質問")
        assert exc.value.code == status
        assert len(requests) == 3


def test_wait_message_uses_timeout_setting(monkeypatch, capsys):
    monkeypatch.setattr(ask, "HTTP_TIMEOUT_SECONDS", 90)
    monkeypatch.setattr(ask, "ask", lambda question: types.GenerateContentResponse())
    monkeypatch.setattr("sys.argv", ["ask", "質問"])
    ask.main()
    output = capsys.readouterr().out
    assert "90秒" in output
    assert "60秒" not in output
