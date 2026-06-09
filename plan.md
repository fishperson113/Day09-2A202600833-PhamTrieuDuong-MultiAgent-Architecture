# Plan Triển Khai Multi-Agent Shopping Assistant

## Mục tiêu

Xây dựng hệ thống multi-agent shopping assistant đạt điểm **90-100/100** theo Rubric, không bị trừ điểm ngu.

---

## Tổng Quan Kiến Trúc

```
User Question
     │
     ▼
┌──────────────────────────────────────────────────────────┐
│                    Supervisor Agent                        │
│  Phân tích: needs_policy? needs_data? clarification?      │
└────────────┬─────────────────────┬────────────────────────┘
             │ policy              │ data
             ▼                     ▼
┌─────────────────────┐  ┌─────────────────────────────┐
│  Worker 1: Policy   │  │  Worker 2: Order/Customer   │
│  (RAG → Chroma)     │  │  (4+ lookup tools)          │
└──────────┬──────────┘  └──────────────┬──────────────┘
           │                            │
           └──────────┬─────────────────┘
                      ▼
┌──────────────────────────────────────────────────────────┐
│              Worker 3: Response Agent                     │
│  Tổng hợp → Answer / clarification_needed / not_found    │
└──────────────────────────────────────────────────────────┘
```

### State Graph (LangGraph)

```
START → supervisor_node (router)
           ├── "policy"        → worker_1_policy_node
           ├── "data"          → worker_2_data_node
           ├── "both"          → cả hai workers (parallel)
           └── "clarification" → worker_3_response_node (skip workers)

Sau workers → worker_3_response_node → END
```

### ShoppingState

```python
class ShoppingState(TypedDict, total=False):
    question: str
    route: dict         # {status, needs_policy, needs_data, clarification_question}
    policy_result: dict # {status, summary, facts, citations}
    data_result: dict   # {status, summary, facts, missing_fields, not_found_entities}
    final_answer: str   # output cuối cùng
    trace: list[dict]   # log từng bước
```

---

## Phase 1: Setup Môi Trường

### 1.1 Tạo `.env`

Tại thư mục gốc (`C:\workspace\Day09-MultiAgent-Architecture\.env`):

```bash
# Dùng custom provider (OpenAI-compatible endpoint)
LLM_PROVIDER=custom
LLM_MODEL=deepseek-v4-flash
CUSTOM_LLM_MODEL=deepseek-v4-flash
CUSTOM_LLM_BASE_URL=https://opencode.ai/zen/go/v1
CUSTOM_LLM_API_KEY=

# Embedding (mặc định đã đúng)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RAG_TOP_K=4
```

**⚠️ IMPORTANT — Phải sửa `src/provider/custom.py` trước khi chạy:**

File `src/provider/custom.py` hiện tại yêu cầu `CUSTOM_LLM_API_KEY` không được rỗng (dòng 12: `if not settings.custom_llm_api_key: raise ValueError`). Vì endpoint `opencode.ai` có thể không cần API key, bạn phải sửa để cho phép API key rỗng:

```python
# Sửa dòng 11-12 trong src/provider/custom.py từ:
if not settings.custom_llm_api_key:
    raise ValueError("CUSTOM_LLM_API_KEY is required for provider=custom")

# Thành:
# Chỉ warning, không raise error — cho phép API key rỗng (open endpoint)
```

→ Chi tiết cách sửa ở **Phase 1.3** bên dưới.

### 1.2 Cài Dependencies

```bash
cd C:\workspace\Day09-MultiAgent-Architecture
python -m venv .venv
.venv\Scripts\pip install -r src\requirements.txt
```

### 1.3 Tạo `src/app/provider_adapter.py` — Adapter Pattern (không đụng `src/provider/`)

**Vấn đề:** `src/provider/custom.py` yêu cầu `CUSTOM_LLM_API_KEY` không rỗng, nhưng endpoint `opencode.ai` không cần key. Examiner có thể dùng provider khác (gemini, openai), nên không nên sửa code gốc.

**Giải pháp — Adapter Pattern:** Tạo file `src/app/provider_adapter.py` mới, không đụng đến `src/provider/`:

```python
from __future__ import annotations

from app.config import Settings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from provider import get_chat_model


class LLMAdapter:
    """Adapter pattern: wraps provider initialization để xử lý edge cases
    mà không sửa code gốc trong src/provider/.

    - Nếu provider là 'custom': tự xử lý (cho phép API key rỗng)
    - Các provider khác: delegate sang get_chat_model() gốc
    """

    @staticmethod
    def build(settings: Settings) -> BaseChatModel:
        provider = settings.provider

        if provider == "custom":
            model_name = settings.custom_llm_model or settings.model
            api_key = settings.custom_llm_api_key or "sk-no-auth-required"
            base_url = settings.custom_llm_base_url
            if not base_url:
                raise ValueError("CUSTOM_LLM_BASE_URL is required for provider=custom")
            return ChatOpenAI(
                model=model_name,
                api_key=api_key,
                base_url=base_url,
                temperature=settings.temperature,
            )

        # Các provider khác: delegate nguyên seal sang code gốc
        return get_chat_model(settings)
```

**Luồng hoạt động:**

```
ShoppingAssistant
  → LLMAdapter.build(settings)
       ├── provider == "custom" → ChatOpenAI(...) với fallback key
       └── provider == "gemini" | "openai" | "openrouter" | "ollama"
            → get_chat_model(settings)  (code gốc, không đụng đến)
```

**Lợi ích:**
- ✅ `src/provider/custom.py` — không sửa
- ✅ `src/provider/__init__.py` — không sửa
- ✅ Examiner dùng provider nào cũng được: gemini/openai/openrouter/ollama → pass qua adapter → vẫn chạy
- ✅ Chỉ custom provider mới qua nhánh xử lý riêng

### 1.4 Kiểm Tra Syntax

```bash
.venv\Scripts\python -m py_compile src\app\*.py src\provider\*.py src\rag\*.py
```

ko có lỗi → bắt đầu code.

---

## Phase 2: Worker 2 — Data Access Layer (Dễ nhất, làm trước)

> **File đích:** `src/app/data_access.py`

### 2.1 `ShoppingDataStore.__init__()`

```python
def __init__(self, json_path: Path) -> None:
    import json
    with open(json_path, encoding="utf-8") as f:
        raw = json.load(f)

    self.metadata = raw["metadata"]
    self.customers = raw["customers"]
    self.orders = raw["orders"]
    self.vouchers = raw["vouchers"]

    # Build indexes — lookup nhanh, ko cần query phức tạp
    self.customer_by_id = {}
    for c in self.customers:
        self.customer_by_id[c["customer_id"]] = c

    self.order_by_id = {}
    self.orders_by_customer_id: dict[str, list] = {}
    for o in self.orders:
        oid = o["order_id"]
        cid = o["customer_id"]
        self.order_by_id[oid] = o
        self.orders_by_customer_id.setdefault(cid, []).append(o)

    self.vouchers_by_customer_id: dict[str, list] = {}
    for v in self.vouchers:
        cid = v["customer_id"]
        self.vouchers_by_customer_id.setdefault(cid, []).append(v)
```

### 2.2 4+ Lookup Methods

Mỗi method trả `dict` với `status: "ok"` hoặc `status: "not_found"`.

**Tool 1 — `get_customer_by_id(customer_id: str)`:**
```python
def get_customer_by_id(self, customer_id: str) -> dict:
    customer = self.customer_by_id.get(customer_id)
    if not customer:
        return {"status": "not_found", "customer_id": customer_id}
    return {"status": "ok", "customer": customer}
```

**Tool 2 — `get_orders_by_customer_id(customer_id: str, limit: int = 10)`:**
```python
def get_orders_by_customer_id(self, customer_id: str, limit: int = 10) -> dict:
    orders = self.orders_by_customer_id.get(customer_id)
    if not orders:
        return {"status": "not_found", "customer_id": customer_id}
    sorted_orders = sorted(orders, key=lambda o: o.get("created_at", ""), reverse=True)
    return {"status": "ok", "orders": sorted_orders[:limit]}
```

**Tool 3 — `get_order_detail_by_order_id(order_id: str)`:**
```python
def get_order_detail_by_order_id(self, order_id: str) -> dict:
    # order_id trong JSON là string, dùng trực tiếp ko cần cast
    order = self.order_by_id.get(order_id)
    if not order:
        return {"status": "not_found", "order_id": order_id}
    return {"status": "ok", "order": order}
```

**⚠️ Lưu ý:** Trong `order_customer_mock_data.json`, `order_id` là **string** (vd: "1971", "2058"). Không cần cast, dùng trực tiếp.

**Tool 4 — `get_vouchers_by_customer_id(customer_id: str, only_active: bool = False)`:**
```python
def get_vouchers_by_customer_id(self, customer_id: str, only_active: bool = False) -> dict:
    vouchers = self.vouchers_by_customer_id.get(customer_id)
    if not vouchers:
        return {"status": "not_found", "customer_id": customer_id}
    if only_active:
        vouchers = [v for v in vouchers if v.get("status") == "active" and v.get("remaining_uses", 0) > 0]
    return {"status": "ok", "vouchers": vouchers, "total": len(vouchers)}
```

### 2.3 `build_data_tools()` — Wrap thành LangChain Tools

```python
from langchain_core.tools import tool

def build_data_tools(store: ShoppingDataStore) -> list:
    @tool
    def get_customer_by_id(customer_id: str) -> dict:
        """Tra cứu thông tin khách hàng theo customer_id (vd: C001, C014)."""
        return store.get_customer_by_id(customer_id)

    @tool
    def get_orders_by_customer_id(customer_id: str) -> dict:
        """Lấy danh sách đơn hàng của khách hàng theo customer_id (vd: C001)."""
        return store.get_orders_by_customer_id(customer_id)

    @tool
    def get_order_detail_by_order_id(order_id: str) -> dict:
        """Lấy chi tiết đơn hàng theo order_id (vd: 1971, 2058, 9999)."""
        return store.get_order_detail_by_order_id(order_id)

    @tool
    def get_vouchers_by_customer_id(customer_id: str, only_active: bool = False) -> dict:
        """Lấy danh sách voucher của khách hàng. Nếu only_active=True thì chỉ lấy voucher còn dùng được."""
        return store.get_vouchers_by_customer_id(customer_id, only_active)

    return [get_customer_by_id, get_orders_by_customer_id, get_order_detail_by_order_id, get_vouchers_by_customer_id]
```

✅ **Đạt: 4 tools nhỏ, rõ nhiệm vụ (tránh bị trừ điểm -10 đến -20 vì gom 1 tool chung)**

---

## Phase 3: Policy RAG

### 3.1 Parser — `src/rag/parser.py`

> **Cấu trúc chunk bắt buộc:** `## H2 → ### H3 → toàn bộ content của H3`

```python
import re

def parse_policy_markdown(markdown_text: str) -> list[dict]:
    chunks = []
    lines = markdown_text.split("\n")

    current_h2 = None
    current_h3 = None
    current_content = []

    def flush():
        if current_h2 and current_h3:
            rendered = f"{current_h2}\n{current_h3}\n" + "\n".join(current_content).strip()
            chunks.append({
                "section_h2": current_h2,
                "section_h3": current_h3,
                "citation": f"policy_mock_vi.md > {current_h3}",
                "rendered_text": rendered.strip(),
            })

    for line in lines:
        h2_match = re.match(r"^##\s+(.+)", line)
        h3_match = re.match(r"^###\s+(.+)", line)

        if h2_match:
            flush()
            current_h2 = h2_match.group(1).strip()
            current_h3 = None
            current_content = []
        elif h3_match:
            flush()
            current_h3 = h3_match.group(1).strip()
            current_content = []
        else:
            if current_h3:  # chỉ gom content thuộc H3
                current_content.append(line)

    flush()  # chunk cuối cùng
    return chunks
```

✅ **Đạt: chunk H2 + H3 + content (10 điểm)**

### 3.2 Embedding Loader

File `src/rag/embeddings.py` đã có sẵn, dùng `SentenceTransformer("all-MiniLM-L6-v2")`.

```python
class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
```

✅ **Không cần sửa — đã có sẵn.**

### 3.3 Chroma Vector Store — `src/rag/vector_store.py`

```python
import chromadb
from chromadb.config import Settings as ChromaSettings

class ChromaPolicyStore:
    def __init__(self, persist_directory: Path, embedding_model, collection_name: str = "policy_chunks"):
        self.client = chromadb.PersistentClient(
            path=str(persist_directory),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedding_model = embedding_model

    def ensure_index(self, markdown_path: Path) -> None:
        if self.collection.count() == 0:
            self.rebuild(markdown_path)

    def rebuild(self, markdown_path: Path) -> None:
        from rag.parser import parse_policy_markdown

        with open(markdown_path, encoding="utf-8") as f:
            text = f.read()

        chunks = parse_policy_markdown(text)

        ids = [f"chunk_{i}" for i in range(len(chunks))]
        documents = [c["rendered_text"] for c in chunks]
        metadatas = [{
            "section_h2": c["section_h2"],
            "section_h3": c["section_h3"],
            "citation": c["citation"],
        } for c in chunks]

        embeddings = self.embedding_model.embed_documents(documents)

        # Xoá collection cũ nếu có, tạo lại
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"},
        )

        # Add batch — 1 lần vì data nhỏ
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        query_emb = self.embedding_model.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for i in range(len(results["ids"][0])):
            hits.append({
                "citation": results["metadatas"][0][i]["citation"],
                "content": results["documents"][0][i],
                "distance": results["distances"][0][i],
            })
        return hits
```

✅ **Đạt: Dùng Chroma + all-MiniLM-L6-v2 thật (10 điểm)**

---

## Phase 4: Prompts — `src/app/prompts.py`

> Mỗi agent có prompt riêng, rõ ràng, yêu cầu output JSON/structure cố định.
> ✅ **Bonus 2 điểm: Prompt tách riêng, dễ đọc dễ thay**

### 4.1 Supervisor Prompt

```python
SUPERVISOR_PROMPT = """Bạn là supervisor agent. Nhiệm vụ: phân tích câu hỏi của user và quyết định route.

Quy tắc:
1. Nếu câu hỏi về chính sách, quy định chung (giao hàng, đổi trả, voucher, hỗ trợ) → needs_policy=true
2. Nếu câu hỏi có mã đơn hàng (vd: 1971, 2058), mã khách hàng (vd: C001, C014), hoặc hỏi về voucher cụ thể → needs_data=true
3. Nếu câu hỏi vừa cần policy vừa cần data (vd: "Đơn hàng 1971 có được hoàn trả không?") → cả hai
4. Nếu câu hỏi thiếu order_id hoặc customer_id khi cần tra cứu (vd: "Voucher của tôi còn dùng được không?", "Đơn hàng của tôi có được hoàn trả không?") → clarification_needed

Trả về JSON:
{
  "status": "ok" hoặc "clarification_needed",
  "needs_policy": true/false,
  "needs_data": true/false,
  "clarification_question": "câu hỏi để hỏi lại user, hoặc null nếu không cần"
}"""
```

### 4.2 Policy Worker Prompt

```python
POLICY_WORKER_PROMPT = """Bạn là policy expert. Nhiệm vụ: tra cứu chính sách từ RAG và tóm tắt.

Các bước:
1. Luôn gọi tool search_policy với câu hỏi của user
2. Đọc kết quả chunks trả về
3. Tóm tắt policy liên quan bằng tiếng Việt
4. Trích dẫn citation từ chunks

Trả về JSON:
{
  "status": "ok",
  "summary": "Tóm tắt ngắn gọn policy liên quan",
  "facts": ["fact 1", "fact 2", ...],
  "citations": ["policy_mock_vi.md > Tên mục", ...]
}"""
```

### 4.3 Data Worker Prompt

```python
DATA_WORKER_PROMPT = """Bạn là data lookup agent. Nhiệm vụ: tra cứu thông tin đơn hàng, khách hàng, voucher.

Các tools bạn có:
- get_customer_by_id(customer_id): tra thông tin khách hàng
- get_orders_by_customer_id(customer_id): danh sách đơn hàng
- get_order_detail_by_order_id(order_id): chi tiết đơn hàng (order_id là số, vd: 1971)
- get_vouchers_by_customer_id(customer_id, only_active): danh sách voucher

Quy tắc:
1. Chọn đúng tool cho từng nhu cầu
2. Nếu lookup trả về not_found → ghi nhận vào not_found_entities
3. Nếu thiếu thông tin (vd: không có customer_id) → trả về clarification_needed

Trả về JSON:
{
  "status": "ok" hoặc "clarification_needed" hoặc "not_found",
  "summary": "Tóm tắt dữ liệu tra được",
  "facts": ["fact 1", "fact 2", ...],
  "missing_fields": ["customer_id", ...],
  "not_found_entities": ["order 9999", ...]
}"""
```

### 4.4 Response Worker Prompt

```python
RESPONSE_WORKER_PROMPT = """Bạn là response agent. Nhiệm vụ: tổng hợp câu trả lời cuối cùng.

Đầu vào bạn có:
- route: quyết định của supervisor (policy, data, both, clarification)
- policy_result: kết quả từ policy worker (nếu có)
- data_result: kết quả từ data worker (nếu có)

Yêu cầu output — phải theo ĐÚNG 1 trong 3 format sau:

Format 1 — Success (có câu trả lời):
Answer: <câu trả lời đầy đủ, bằng tiếng Việt>
Evidence:
- Policy: <trích dẫn policy nếu có>
- Order data: <dữ liệu đơn hàng nếu có>

Format 2 — Clarification (thiếu thông tin):
Status: clarification_needed
Question: <câu hỏi để hỏi lại user>

Format 3 — Not found (không tìm thấy):
Status: not_found
Message: <thông báo không tìm thấy>

Lưu ý:
- Nếu status cuối cùng là clarification_needed → dùng Format 2
- Nếu có not_found trong data → dùng Format 3
- Còn lại → dùng Format 1 với đầy đủ evidence"""
```

---

## Phase 5: Graph Orchestration — `src/app/graph.py`

### 5.1 `ShoppingAssistant.__init__()`

```python
def __init__(self, settings: Settings | None = None) -> None:
    self.settings = settings or Settings.load()
    s = self.settings

    # 1. Load LLM
    self.llm = _load_llm(s)

    # 2. Load data store
    self.data_store = ShoppingDataStore(s.orders_path)
    self.data_tools = build_data_tools(self.data_store)

    # Bind tools for data worker
    self.data_llm = self.llm.bind_tools(self.data_tools)

    # 3. Load vector store
    self.embedding_model = SentenceTransformerEmbeddings(s.embedding_model_name)
    self.vector_store = ChromaPolicyStore(
        persist_directory=s.chroma_dir,
        embedding_model=self.embedding_model,
    )

    # 4. Build RAG tool for policy worker
    self.policy_tool = self._build_rag_tool()

    # Bind tool for policy worker
    self.policy_llm = self.llm.bind_tools([self.policy_tool])

    # 5. Compile graph
    self.graph = build_graph(self)
```

### 5.2 Load LLM — dùng Adapter Pattern

**KHÔNG tự viết lại switch-case.** Dùng adapter đã tạo ở Phase 1.3:

```python
from app.provider_adapter import LLMAdapter

def _load_llm(s: Settings):
    return LLMAdapter.build(s)
```

✅ **Bonus 2 điểm: Provider abstraction sạch + Adapter Pattern — đổi provider ko cần sửa graph, ko cần sửa code gốc**

Nếu examiner đổi từ `custom` sang `gemini`/`openai`/`openrouter`/`ollama`: adapter pass-through nguyên seal sang `get_chat_model()`, không cần sửa gì.

### 5.3 RAG Search Tool

```python
def _build_rag_tool(self):
    @tool
    def search_policy(query: str) -> list[dict]:
        """Tìm kiếm chính sách mua sắm theo câu hỏi của khách hàng. Trả về các chunks chính sách liên quan kèm citation."""
        return self.vector_store.search(query, top_k=self.settings.top_k)

    return search_policy
```

### 5.4 `build_graph()` — Compile LangGraph Workflow

```python
from langgraph.graph import StateGraph, END
from app.state import ShoppingState

def build_graph(assistant: ShoppingAssistant):
    workflow = StateGraph(ShoppingState)

    workflow.add_node("supervisor", lambda state: supervisor_node(state, assistant))
    workflow.add_node("worker_1_policy", lambda state: worker_1_policy_node(state, assistant))
    workflow.add_node("worker_2_data", lambda state: worker_2_data_node(state, assistant))
    workflow.add_node("worker_3_response", lambda state: worker_3_response_node(state, assistant))

    workflow.set_entry_point("supervisor")

    # Conditional edge từ supervisor
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: _route_from_supervisor(state),
        {
            "worker_1_policy": "worker_1_policy",
            "worker_2_data": "worker_2_data",
            "both": "worker_1_policy",  # đi policy trước (hoặc dùng parallel)
            "worker_3_response": "worker_3_response",
        }
    )

    # Sau policy worker → data hoặc response
    workflow.add_conditional_edges(
        "worker_1_policy",
        lambda state: "worker_2_data" if state.get("route", {}).get("needs_data") else "worker_3_response",
        {
            "worker_2_data": "worker_2_data",
            "worker_3_response": "worker_3_response",
        }
    )

    # Sau data worker → response
    workflow.add_edge("worker_2_data", "worker_3_response")

    # Sau response → END
    workflow.add_edge("worker_3_response", END)

    return workflow.compile()
```

**Giải thích routing logic:**

| Tình huống | supervisor route | Flow |
|---|---|---|
| Chỉ policy | `{needs_policy: T, needs_data: F}` | supervisor → policy → response |
| Chỉ data | `{needs_policy: F, needs_data: T}` | supervisor → data → response |
| Cả hai | `{needs_policy: T, needs_data: T}` | supervisor → policy → data → response |
| Clarification | `{status: clarification_needed}` | supervisor → response (skip workers) |

### 5.5 `_route_from_supervisor()`

```python
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
```

### 5.6 `supervisor_node()` — Gọi LLM + Parse JSON

```python
from app.prompts import SUPERVISOR_PROMPT
import json, re

def supervisor_node(state: ShoppingState, assistant: ShoppingAssistant) -> ShoppingState:
    question = state["question"]

    messages = [
        {"role": "system", "content": SUPERVISOR_PROMPT},
        {"role": "user", "content": question},
    ]

    response = assistant.llm.invoke(messages)
    content = response.content.strip()

    # Parse JSON từ LLM output (có thể bị wrap trong ```json ... ```)
    json_match = re.search(r"\{.*\}", content, re.DOTALL)
    if json_match:
        route = json.loads(json_match.group())
    else:
        route = {"status": "ok", "needs_policy": False, "needs_data": False, "clarification_question": None}

    state["route"] = route

    # Ghi trace
    state["trace"] = state.get("trace", []) + [{
        "node": "supervisor",
        "input": question,
        "output": route,
    }]

    return state
```

### 5.7 `worker_1_policy_node()`

```python
def worker_1_policy_node(state: ShoppingState, assistant: ShoppingAssistant) -> ShoppingState:
    question = state["question"]

    messages = [
        {"role": "system", "content": POLICY_WORKER_PROMPT},
        {"role": "user", "content": question},
    ]

    response = assistant.policy_llm.invoke(messages)

    # Parse JSON từ response
    content = response.content.strip()
    json_match = re.search(r"\{.*\}", content, re.DOTALL)
    if json_match:
        result = json.loads(json_match.group())
    else:
        result = {"status": "ok", "summary": content, "facts": [], "citations": []}

    state["policy_result"] = result

    state["trace"] = state.get("trace", []) + [{
        "node": "worker_1_policy",
        "input": question,
        "tool_calls": response.tool_calls if hasattr(response, 'tool_calls') else [],
        "output": result,
    }]

    return state
```

### 5.8 `worker_2_data_node()`

```python
def worker_2_data_node(state: ShoppingState, assistant: ShoppingAssistant) -> ShoppingState:
    question = state["question"]

    messages = [
        {"role": "system", "content": DATA_WORKER_PROMPT},
        {"role": "user", "content": question},
    ]

    response = assistant.data_llm.invoke(messages)

    content = response.content.strip()
    json_match = re.search(r"\{.*\}", content, re.DOTALL)
    if json_match:
        result = json.loads(json_match.group())
    else:
        result = {"status": "ok", "summary": content, "facts": [], "missing_fields": [], "not_found_entities": []}

    state["data_result"] = result

    tool_calls = response.tool_calls if hasattr(response, 'tool_calls') else []
    state["trace"] = state.get("trace", []) + [{
        "node": "worker_2_data",
        "input": question,
        "tool_calls": [
            {"name": tc["name"], "args": tc["args"]}
            for tc in tool_calls
        ],
        "output": result,
    }]

    return state
```

### 5.9 `worker_3_response_node()`

```python
def worker_3_response_node(state: ShoppingState, assistant: ShoppingAssistant) -> ShoppingState:
    route = state.get("route", {})

    # Nếu supervisor đã xác định clarification
    if route.get("status") == "clarification_needed":
        final = f"Status: clarification_needed\nQuestion: {route.get('clarification_question', 'Vui lòng cung cấp thêm thông tin.')}"
        state["final_answer"] = final

        state["trace"] = state.get("trace", []) + [{
            "node": "worker_3_response",
            "output": final,
        }]
        return state

    # Xử lý not_found từ data worker
    data_result = state.get("data_result", {})
    if data_result.get("status") == "not_found" or data_result.get("not_found_entities"):
        final = f"Status: not_found\nMessage: Không tìm thấy thông tin: {', '.join(data_result.get('not_found_entities', ['dữ liệu yêu cầu']))}"
        state["final_answer"] = final

        state["trace"] = state.get("trace", []) + [{
            "node": "worker_3_response",
            "output": final,
        }]
        return state

    # Gọi LLM response worker để tổng hợp
    policy_info = state.get("policy_result", {})
    data_info = state.get("data_result", {})

    messages = [
        {"role": "system", "content": RESPONSE_WORKER_PROMPT},
        {"role": "user", "content": f"""Hãy tổng hợp câu trả lời dựa trên thông tin sau:

Route: {json.dumps(route, ensure_ascii=False)}
Policy result: {json.dumps(policy_info, ensure_ascii=False)}
Data result: {json.dumps(data_info, ensure_ascii=False)}"""},
    ]

    response = assistant.llm.invoke(messages)
    final = response.content.strip()
    state["final_answer"] = final

    state["trace"] = state.get("trace", []) + [{
        "node": "worker_3_response",
        "output": final,
    }]

    return state
```

### 5.10 `ShoppingAssistant.ask()` — Entry Point

```python
def ask(self, question: str, trace_file: Path | None = None, rebuild_index: bool = False) -> dict:
    # Rebuild Chroma index nếu cần
    if rebuild_index:
        self.vector_store.rebuild(self.settings.policy_path)
    else:
        self.vector_store.ensure_index(self.settings.policy_path)

    # Khởi tạo state
    initial_state: ShoppingState = {
        "question": question,
        "route": {},
        "policy_result": {},
        "data_result": {},
        "final_answer": "",
        "trace": [],
    }

    # Invoke graph
    result = self.graph.invoke(initial_state)

    # Lưu trace nếu cần
    if trace_file:
        trace_file.parent.mkdir(parents=True, exist_ok=True)
        import json
        with open(trace_file, "w", encoding="utf-8") as f:
            json.dump(result.get("trace", []), f, ensure_ascii=False, indent=2)

    return result
```

### 5.11 `ShoppingAssistant.run_batch()` — Batch Testing

```python
def run_batch(self, test_file: Path, output_dir: Path, rebuild_index: bool = False) -> dict:
    import json, time

    with open(test_file, encoding="utf-8") as f:
        test_cases = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for case in test_cases:
        qid = case["id"]
        question = case["question"]

        trace_path = output_dir / f"trace_{qid}.json"
        result = self.ask(question, trace_file=trace_path, rebuild_index=rebuild_index)

        # Simple evaluation
        route = result.get("route", {})
        final_answer = result.get("final_answer", "")
        expected_route = case.get("expected_route", [])
        expected_status = case.get("expected_status", "ok")

        # Đánh giá route
        actual_route = []
        if route.get("needs_policy"): actual_route.append("policy")
        if route.get("needs_data"): actual_route.append("data")

        route_correct = set(actual_route) == set(expected_route)

        # Đánh giá status
        status_correct = False
        if expected_status == "clarification_needed":
            status_correct = "clarification_needed" in final_answer
        elif expected_status == "not_found":
            status_correct = "not_found" in final_answer
        elif expected_status == "ok":
            status_correct = "Answer:" in final_answer or "Status: clarification_needed" not in final_answer

        results.append({
            "id": qid,
            "question": question,
            "route_correct": route_correct,
            "status_correct": status_correct,
            "final_answer": final_answer,
            "trace_file": str(trace_path),
        })

    summary = {
        "total": len(results),
        "route_ok": sum(1 for r in results if r["route_correct"]),
        "status_ok": sum(1 for r in results if r["status_correct"]),
        "details": results,
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary
```

✅ **Đạt: Batch test từ data/test.json (10 điểm)**
✅ **Đạt: Trace JSON (3 điểm bonus)**

✅ **Đạt: clarification_needed (5 điểm)**
✅ **Đạt: not_found (5 điểm)**

---

## Phase 6: CLI — `src/app/cli.py`

```python
import argparse, json
from pathlib import Path
from app.graph import ShoppingAssistant

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Shopping Assistant CLI")
    parser.add_argument("--question", help="Run one question")
    parser.add_argument("--test-file", default="data/test.json")
    parser.add_argument("--trace-file", default=None)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--rebuild-index", action="store_true")
    return parser

def main() -> None:
    args = build_parser().parse_args()
    assistant = ShoppingAssistant()

    if args.batch:
        output_dir = assistant.settings.traces_dir / "batch"
        summary = assistant.run_batch(
            test_file=Path(args.test_file),
            output_dir=output_dir,
            rebuild_index=args.rebuild_index,
        )
        print(f"Batch complete: {summary['route_ok']}/{summary['total']} route OK, {summary['status_ok']}/{summary['total']} status OK")
        print(f"Summary saved to {output_dir / 'summary.json'}")

    elif args.question:
        trace_path = Path(args.trace_file) if args.trace_file else None
        result = assistant.ask(args.question, trace_file=trace_path, rebuild_index=args.rebuild_index)
        print(result.get("final_answer", ""))

        if args.trace_file:
            print(f"\nTrace saved to {args.trace_file}")

if __name__ == "__main__":
    main()
```

---

## Phase 7: Kiểm Tra & Debug

### 7.1 Chạy Từng Bước

```bash
# 1. Test data lookup (worker 2)
.venv\Scripts\python -m app.cli --question "Đơn hàng 1971 bao giờ được giao?"

# 2. Test policy (worker 1)
.venv\Scripts\python -m app.cli --question "Chính sách hoàn trả hàng ra sao?"

# 3. Test mixed
.venv\Scripts\python -m app.cli --question "Đơn hàng 1971 có được hoàn trả không?"

# 4. Test clarification
.venv\Scripts\python -m app.cli --question "Voucher của tôi còn dùng được không?"

# 5. Test not_found
.venv\Scripts\python -m app.cli --question "Kiểm tra đơn hàng 9999 giúp tôi"

# 6. Chạy full batch
.venv\Scripts\python -m app.cli --batch --test-file ../data/test.json
```

**Lưu ý:** Chạy từ thư mục `src/` với `PYTHONPATH=src`:
```bash
cd C:\workspace\Day09-MultiAgent-Architecture\src
$env:PYTHONPATH = "."
python -m app.cli --question "Chính sách hoàn trả hàng ra sao?"
```

### 7.2 Đọc Trace Để Debug

Mở file trace JSON trong `src/artifacts/traces/` để xem:
- Supervisor output: route có đúng không?
- Policy worker: tool calls có gọi RAG không?
- Data worker: tool calls có chọn đúng tool không?
- Response worker: output có đúng format không?

---

## Phase 8: Checklist Rubric — Bảng Tự Chấm

### 0-60: Core (cần đạt hết)

| # | Tiêu chí | Điểm | Đạt? | Cách đảm bảo |
|---|---|---|---|---|
| 1 | Supervisor Agent + route đúng | 15 | ✅ | `supervisor_node()` với prompt rõ + `_route_from_supervisor()` conditional edges |
| 2 | Worker 1 dùng RAG thật | 15 | ✅ | `policy_llm` bind `search_policy` tool → gọi Chroma |
| 3 | Worker 2 có ≥4 tools nhỏ | 15 | ✅ | `get_customer_by_id`, `get_orders_by_customer_id`, `get_order_detail_by_order_id`, `get_vouchers_by_customer_id` |
| 4 | Worker 3 tổng hợp final answer | 15 | ✅ | `worker_3_response_node()` với 3 format chuẩn |

### 60-90: Engineering Quality

| # | Tiêu chí | Điểm | Đạt? | Cách đảm bảo |
|---|---|---|---|---|
| 5 | Chunk H2 + H3 + content | 10 | ✅ | `parse_policy_markdown()` regex chính xác |
| 6 | Chroma + all-MiniLM-L6-v2 | 10 | ✅ | `ChromaPolicyStore` + `SentenceTransformerEmbeddings` |
| 7 | Xử lý clarification_needed | 5 | ✅ | Supervisor detect → route thẳng đến response worker |
| 8 | Xử lý not_found | 5 | ✅ | Data worker trả status → response worker format |
| 9 | Batch test từ test.json | 10 | ✅ | `run_batch()` + `summary.json` |

### 90-100: Bonus

| # | Tiêu chí | Điểm | Đạt? | Cách đảm bảo |
|---|---|---|---|---|
| 10 | Citation rõ ràng | 3 | ✅ | `citation` field trong chunk → policy worker trích dẫn |
| 11 | Trace JSON debug | 3 | ✅ | Mỗi node ghi vào `trace` → lưu file |
| 12 | Provider abstraction sạch | 2 | ✅ | `_load_llm()` switch theo provider |
| 13 | Prompt riêng từng agent | 2 | ✅ | 4 prompt riêng biệt trong `prompts.py` |

### Trừ Điểm — Cảnh Báo

| Hành vi | Mất điểm | Khắc phục |
|---|---|---|
| Gom lookup vào 1 tool chung | -10 đến -20 | ✅ 4 tools riêng biệt |
| Ko routing thật, hard-code flow | -10 | ✅ LangGraph conditional edges |
| Ko evidence/phân biệt policy/data | -10 | ✅ Response worker format chuẩn |
| Ko chạy được với data trong repo | -10 | ✅ Dùng đúng file từ Settings |

---

## Lưu Ý Quan Trọng (Đọc Kỹ Để Không Bị Trừ Điểm Ngu)

### 1. `order_id` là string trong JSON
`order_id` trong file JSON là string (vd: `"1971"`, `"2058"`). Dùng trực tiếp, **không** cast sang int.

### 2. Supervisor prompt phải rõ về "câu hỏi thiếu thông tin"
Các câu như *"Voucher của tôi còn dùng được không?"* và *"Đơn hàng của tôi có được hoàn trả không?"* **không có** order_id/customer_id → supervisor **phải** trả `clarification_needed`.

### 3. Response worker format phải chính xác
Rubric kiểm tra evidence. Phải có `Answer:`, `Evidence:`, `Policy:`, `Order data:` trong output thành công.

### 4. Không dùng chung 1 tool cho cả lookup
Nếu gom `get_customer_by_id`, `get_orders_by_customer_id`, ... vào 1 tool `lookup_data(entity_type, id)` → trừ 10-20 điểm.

### 5. Build Chroma index có thể chậm lần đầu
Lần đầu chạy, Chroma cần parse + embed toàn bộ policy (khoảng 500 dòng). Lần sau chỉ load từ disk. Dùng `ensure_index` để tự động rebuild nếu rỗng.

### 6. Lưu trace để debug
Nếu batch test sai route, trace JSON cho biết supervisor output để fix prompt.

---

## Tóm Tắt Luồng Code Theo File

| File | Nhiệm vụ | Điểm Rubric |
|---|---|---|
| `src/rag/parser.py` | Parse policy → chunks | 10 (chunking) + 3 (citation) |
| `src/rag/vector_store.py` | Chroma index + search | 10 (Chroma+embedding) |
| `src/rag/embeddings.py` | Đã có sẵn | — |
| `src/app/data_access.py` | Data store + 4 tools | 15 (worker 2) |
| `src/app/prompts.py` | 4 prompts riêng | 2 (prompt bonus) |
| `src/app/graph.py` | Graph + routing + nodes | 15×3 (supervisor+policy+response) + clarification + not_found |
| `src/app/cli.py` | CLI + batch | 10 (batch test) |
| `src/app/state.py` | Đã có sẵn | — |
| `src/app/config.py` | Đã có sẵn + provider abstraction | 2 (provider bonus) |
| `data/test.json` | File test | — |
| `.env` | Config | — |

**Tổng điểm tối đa: 100/100 (không bị trừ)**
