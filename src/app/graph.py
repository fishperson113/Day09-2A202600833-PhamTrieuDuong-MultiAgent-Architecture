from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from langchain_core.tools import tool
from langgraph.graph import END, StateGraph

from app.config import Settings
from app.data_access import ShoppingDataStore, build_data_tools
from app.provider_adapter import LLMAdapter
from app.prompts import (
    DATA_WORKER_PROMPT,
    POLICY_WORKER_PROMPT,
    RESPONSE_WORKER_PROMPT,
    SUPERVISOR_PROMPT,
)
from app.state import ShoppingState
from rag.embeddings import SentenceTransformerEmbeddings
from rag.vector_store import ChromaPolicyStore


class ShoppingAssistant:
    """Multi-agent shopping assistant graph."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.load()
        s = self.settings

        # 1. Load LLM
        self.llm = LLMAdapter.build(s)

        # 2. Load data store + tools
        self.data_store = ShoppingDataStore(s.orders_path)
        self.data_tools = build_data_tools(self.data_store)
        self.data_llm = self.llm.bind_tools(self.data_tools)

        # 3. Load embedding + vector store
        self.embedding_model = SentenceTransformerEmbeddings(s.embedding_model_name)
        self.vector_store = ChromaPolicyStore(
            persist_directory=s.chroma_dir,
            embedding_model=self.embedding_model,
        )

        # 4. Build RAG tool
        self.policy_tool = self._build_rag_tool()
        self.policy_llm = self.llm.bind_tools([self.policy_tool])

        # 5. Compile graph
        self.graph = self._build_graph()

    # ------------------------------------------------------------------
    # RAG tool
    # ------------------------------------------------------------------

    def _build_rag_tool(self):
        @tool
        def search_policy(query: str) -> list[dict[str, Any]]:
            """Tìm kiếm chính sách mua sắm theo câu hỏi của khách hàng. Trả về các chunks chính sách liên quan kèm citation."""
            return self.vector_store.search(query, top_k=self.settings.top_k)

        return search_policy

    # ------------------------------------------------------------------
    # Graph definition
    # ------------------------------------------------------------------

    def _build_graph(self):
        workflow = StateGraph(ShoppingState)

        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("worker_1_policy", self._worker_1_policy_node)
        workflow.add_node("worker_2_data", self._worker_2_data_node)
        workflow.add_node("worker_3_response", self._worker_3_response_node)

        workflow.set_entry_point("supervisor")

        # Supervisor → conditional route
        workflow.add_conditional_edges(
            "supervisor",
            _route_from_supervisor,
            {
                "worker_1_policy": "worker_1_policy",
                "worker_2_data": "worker_2_data",
                "both": "worker_1_policy",  # policy trước, data sau
                "worker_3_response": "worker_3_response",
            },
        )

        # Sau policy worker → data worker (nếu cần) hoặc response
        workflow.add_conditional_edges(
            "worker_1_policy",
            _route_after_policy,
            {
                "worker_2_data": "worker_2_data",
                "worker_3_response": "worker_3_response",
            },
        )

        # Sau data worker → response
        workflow.add_edge("worker_2_data", "worker_3_response")

        # Sau response → END
        workflow.add_edge("worker_3_response", END)

        return workflow.compile()

    # ------------------------------------------------------------------
    # Node implementations
    # ------------------------------------------------------------------

    def _supervisor_node(self, state: ShoppingState) -> ShoppingState:
        question = state["question"]

        response = self.llm.invoke([
            {"role": "system", "content": SUPERVISOR_PROMPT},
            {"role": "user", "content": question},
        ])
        content = response.content.strip()

        route = _parse_json_from_llm(content) or {
            "status": "ok",
            "needs_policy": False,
            "needs_data": False,
            "clarification_question": None,
        }

        state["route"] = route
        _append_trace(state, {
            "node": "supervisor",
            "input": question,
            "output": route,
        })
        return state

    def _worker_1_policy_node(self, state: ShoppingState) -> ShoppingState:
        question = state["question"]

        response = self.policy_llm.invoke([
            {"role": "system", "content": POLICY_WORKER_PROMPT},
            {"role": "user", "content": question},
        ])
        content = response.content.strip()

        result = _parse_json_from_llm(content) or {
            "status": "ok",
            "summary": content,
            "facts": [],
            "citations": [],
        }

        state["policy_result"] = result
        _append_trace(state, {
            "node": "worker_1_policy",
            "input": question,
            "tool_calls": _extract_tool_calls(response),
            "output": result,
        })
        return state

    def _worker_2_data_node(self, state: ShoppingState) -> ShoppingState:
        question = state["question"]

        response = self.data_llm.invoke([
            {"role": "system", "content": DATA_WORKER_PROMPT},
            {"role": "user", "content": question},
        ])
        content = response.content.strip()

        result = _parse_json_from_llm(content) or {
            "status": "ok",
            "summary": content,
            "facts": [],
            "missing_fields": [],
            "not_found_entities": [],
        }

        # Fallback: nếu tool calls có not_found nhưng LLM ko parse được
        # (LLM yếu, ignore tool result), tự detect từ tool_calls
        tool_calls = _extract_tool_calls(response)
        if not result.get("not_found_entities") and not result.get("missing_fields"):
            for tc in tool_calls:
                name = tc["name"]
                args = tc["args"]
                # Re-execute tool để check result
                for t in self.data_tools:
                    if t.name == name:
                        tool_result = t.invoke(args)
                        if isinstance(tool_result, dict) and tool_result.get("status") == "not_found":
                            entity = f"{name.split('_by_')[0] if '_by_' in name else name} {args.get('order_id') or args.get('customer_id') or ''}"
                            result["not_found_entities"].append(entity.strip())
                            result["status"] = "not_found"
                        break

        state["data_result"] = result
        _append_trace(state, {
            "node": "worker_2_data",
            "input": question,
            "tool_calls": _extract_tool_calls(response),
            "output": result,
        })
        return state

    def _worker_3_response_node(self, state: ShoppingState) -> ShoppingState:
        route = state.get("route", {})

        # Case 1: supervisor đã bảo clarification
        if route.get("status") == "clarification_needed":
            final = (
                f"Status: clarification_needed\n"
                f"Question: {route.get('clarification_question', 'Vui lòng cung cấp thêm thông tin.')}"
            )
            state["final_answer"] = final
            _append_trace(state, {"node": "worker_3_response", "output": final})
            return state

        # Case 2: not_found từ data worker
        data_result = state.get("data_result", {})
        not_found = data_result.get("not_found_entities", []) or (
            [data_result.get("summary", "dữ liệu yêu cầu")]
            if data_result.get("status") == "not_found"
            else []
        )
        if not_found:
            final = f"Status: not_found\nMessage: Không tìm thấy thông tin: {', '.join(not_found)}"
            state["final_answer"] = final
            _append_trace(state, {"node": "worker_3_response", "output": final})
            return state

        # Case 3: tổng hợp bằng LLM
        policy_info = state.get("policy_result", {})
        data_info = state.get("data_result", {})

        response = self.llm.invoke([
            {"role": "system", "content": RESPONSE_WORKER_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Hãy tổng hợp câu trả lời dựa trên thông tin sau:\n\n"
                    f"Route: {json.dumps(route, ensure_ascii=False)}\n"
                    f"Policy result: {json.dumps(policy_info, ensure_ascii=False)}\n"
                    f"Data result: {json.dumps(data_info, ensure_ascii=False)}"
                ),
            },
        ])

        final = response.content.strip()
        state["final_answer"] = final
        _append_trace(state, {"node": "worker_3_response", "output": final})
        return state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(
        self,
        question: str,
        trace_file: Path | None = None,
        rebuild_index: bool = False,
    ) -> dict[str, Any]:
        """Chạy một câu hỏi qua graph."""
        if rebuild_index:
            self.vector_store.rebuild(self.settings.policy_path)
        else:
            self.vector_store.ensure_index(self.settings.policy_path)

        initial_state: ShoppingState = {
            "question": question,
            "route": {},
            "policy_result": {},
            "data_result": {},
            "final_answer": "",
            "trace": [],
        }

        result = self.graph.invoke(initial_state)

        if trace_file:
            trace_file.parent.mkdir(parents=True, exist_ok=True)
            with open(trace_file, "w", encoding="utf-8") as f:
                json.dump(result.get("trace", []), f, ensure_ascii=False, indent=2)

        return result

    def run_batch(
        self,
        test_file: Path,
        output_dir: Path,
        rebuild_index: bool = False,
    ) -> dict[str, Any]:
        """Chạy batch test từ test.json và lưu summary + traces."""
        with open(test_file, encoding="utf-8") as f:
            test_cases = json.load(f)

        output_dir.mkdir(parents=True, exist_ok=True)
        details = []

        for case in test_cases:
            qid = case["id"]
            question = case["question"]
            trace_path = output_dir / f"trace_{qid}.json"

            t0 = time.time()
            result = self.ask(question, trace_file=trace_path, rebuild_index=rebuild_index)
            elapsed = time.time() - t0

            route = result.get("route", {})
            final_answer = result.get("final_answer", "")
            expected_route = case.get("expected_route", [])
            expected_status = case.get("expected_status", "ok")

            # Evaluate route
            actual_route = []
            if route.get("needs_policy"):
                actual_route.append("policy")
            if route.get("needs_data"):
                actual_route.append("data")
            route_correct = set(actual_route) == set(expected_route)

            # Evaluate status
            status_correct = _check_status(final_answer, expected_status)

            details.append({
                "id": qid,
                "question": question,
                "route_actual": actual_route,
                "route_expected": expected_route,
                "route_correct": route_correct,
                "status_correct": status_correct,
                "elapsed": round(elapsed, 2),
                "trace_file": str(trace_path),
            })

        summary = {
            "total": len(details),
            "route_ok": sum(1 for d in details if d["route_correct"]),
            "status_ok": sum(1 for d in details if d["status_correct"]),
            "details": details,
        }

        summary_path = output_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        return summary


# ======================================================================
# Helpers
# ======================================================================


def _route_from_supervisor(state: ShoppingState) -> str:
    route = state.get("route", {})
    if route.get("status") == "clarification_needed":
        return "worker_3_response"
    needs_policy = route.get("needs_policy", False)
    needs_data = route.get("needs_data", False)
    if needs_policy and needs_data:
        return "both"
    if needs_policy:
        return "worker_1_policy"
    if needs_data:
        return "worker_2_data"
    return "worker_3_response"


def _route_after_policy(state: ShoppingState) -> str:
    route = state.get("route", {})
    needs_data = route.get("needs_data", False)
    if needs_data:
        return "worker_2_data"
    return "worker_3_response"


def _parse_json_from_llm(content: str) -> dict | None:
    """Trích xuất JSON object từ LLM output (có thể bị wrap trong ```json ... ```)."""
    json_match = re.search(r"\{.*\}", content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            return None
    return None


def _extract_tool_calls(response: Any) -> list[dict]:
    """Trích xuất tool calls từ LLM response để ghi trace."""
    if hasattr(response, "tool_calls") and response.tool_calls:
        return [
            {"name": tc["name"], "args": tc["args"]}
            for tc in response.tool_calls
        ]
    return []


def _append_trace(state: ShoppingState, entry: dict) -> None:
    if "trace" not in state:
        state["trace"] = []
    state["trace"].append(entry)


def _check_status(final_answer: str, expected_status: str) -> bool:
    if expected_status == "clarification_needed":
        return "clarification_needed" in final_answer
    if expected_status == "not_found":
        return "not_found" in final_answer
    # expected_status == "ok": phải có Answer: và ko phải lỗi
    return "Answer:" in final_answer


# ======================================================================
# build_graph standalone (giữ nguyên signature cho CLI)
# ======================================================================


def build_graph() -> Any:
    """Standalone graph builder (backward-compatible stub)."""
    raise NotImplementedError("Dùng ShoppingAssistant() thay vì build_graph()")
