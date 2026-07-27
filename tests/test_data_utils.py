"""Unit tests for src/data_utils.py -- the single source of truth for the prompt
format. The README calls out train/eval prompt drift as a key pitfall, so these
tests lock the prompt shape and the JSONL loader's behavior in place.
"""
import pytest

from src.data_utils import (
    SCHEMA_SQL,
    SYSTEM_PROMPT,
    build_messages,
    build_plain_prompt,
    build_user_prompt,
    load_jsonl,
)

QUESTION = "How many departments are located in London?"


class TestPromptShape:
    def test_user_prompt_contains_schema_question_and_cue(self):
        prompt = build_user_prompt(QUESTION)
        assert SCHEMA_SQL in prompt
        assert QUESTION in prompt
        assert prompt.rstrip().endswith("SQL:")

    def test_messages_are_system_then_user(self):
        msgs = build_messages(QUESTION)
        assert [m["role"] for m in msgs] == ["system", "user"]
        assert msgs[0]["content"] == SYSTEM_PROMPT
        assert QUESTION in msgs[1]["content"]

    def test_plain_prompt_embeds_system_and_user(self):
        plain = build_plain_prompt(QUESTION)
        assert SYSTEM_PROMPT in plain
        assert build_user_prompt(QUESTION) in plain

    def test_chat_and_plain_share_the_same_user_body(self):
        # Parity guard: the user-visible schema+question text must be identical
        # whether we go through the chat template or the plain fallback.
        assert build_messages(QUESTION)[1]["content"] in build_plain_prompt(QUESTION)


class TestLoadJsonl:
    def test_round_trip_and_skips_blank_lines(self, tmp_path):
        p = tmp_path / "rows.jsonl"
        p.write_text(
            '{"id": 1, "question": "q1", "sql": "SELECT 1"}\n'
            "\n"
            '  \n'
            '{"id": 2, "question": "q2", "sql": "SELECT 2"}\n',
            encoding="utf-8",
        )
        rows = load_jsonl(p)
        assert len(rows) == 2
        assert rows[0] == {"id": 1, "question": "q1", "sql": "SELECT 1"}
        assert rows[1]["id"] == 2

    def test_invalid_json_raises_valueerror_with_line_number(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        p.write_text('{"id": 1}\n{not valid json}\n', encoding="utf-8")
        with pytest.raises(ValueError) as exc:
            load_jsonl(p)
        assert "bad.jsonl:2" in str(exc.value)

    def test_accepts_str_and_path(self, tmp_path):
        p = tmp_path / "one.jsonl"
        p.write_text('{"id": 1}\n', encoding="utf-8")
        assert load_jsonl(str(p)) == load_jsonl(p) == [{"id": 1}]
