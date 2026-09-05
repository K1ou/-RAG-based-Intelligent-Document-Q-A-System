from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ConversationTurn:
    question: str
    answer: str


class InMemoryConversationStore:
    def __init__(self) -> None:
        self._data: dict[str, list[ConversationTurn]] = {}

    def append_turn(self, session_id: str, question: str, answer: str) -> None:
        if not session_id.strip():
            raise ValueError("session_id must not be empty")
        turns = self._data.setdefault(session_id, [])
        turns.append(ConversationTurn(question=question, answer=answer))

    def get_recent_turns(self, session_id: str, limit: int = 4) -> list[ConversationTurn]:
        if limit <= 0:
            raise ValueError("limit must be greater than 0")
        turns = self._data.get(session_id, [])
        return turns[-limit:]

    def to_dict(self) -> dict[str, list[dict[str, str]]]:
        return {
            session_id: [{"question": t.question, "answer": t.answer} for t in turns]
            for session_id, turns in self._data.items()
        }

    def load_dict(self, payload: dict[str, list[dict[str, str]]]) -> None:
        self._data = {}
        for session_id, rows in payload.items():
            for row in rows:
                self.append_turn(
                    session_id=session_id,
                    question=str(row.get("question", "")),
                    answer=str(row.get("answer", "")),
                )


def build_contextualized_query(question: str, history_turns: list[ConversationTurn], max_turns: int = 4) -> str:
    q = question.strip()
    if not q:
        raise ValueError("question must not be empty")
    if not history_turns:
        return q

    recent = history_turns[-max_turns:]
    history_lines: list[str] = []
    for idx, turn in enumerate(recent, start=1):
        history_lines.append(f"第{idx}轮 用户: {turn.question}")
        history_lines.append(f"第{idx}轮 助手: {turn.answer}")

    history_block = "\n".join(history_lines)
    return (
        "请结合历史对话理解当前问题中的代词、省略和上下文。\n"
        f"历史对话:\n{history_block}\n"
        f"当前问题: {q}"
    )
