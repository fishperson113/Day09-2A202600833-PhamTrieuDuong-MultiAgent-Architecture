from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.tools import tool


class ShoppingDataStore:
    """Mock-data lookup với index trong memory."""

    def __init__(self, json_path: Path) -> None:
        with open(json_path, encoding="utf-8") as f:
            raw = json.load(f)

        self.metadata = raw["metadata"]
        self.customers = raw["customers"]
        self.orders = raw["orders"]
        self.vouchers = raw["vouchers"]

        # Build indexes
        self.customer_by_id: dict[str, dict] = {}
        for c in self.customers:
            self.customer_by_id[c["customer_id"]] = c

        self.order_by_id: dict[int, dict] = {}
        self.orders_by_customer_id: dict[str, list[dict]] = {}
        for o in self.orders:
            oid = o["order_id"]
            cid = o["customer_id"]
            self.order_by_id[oid] = o
            self.orders_by_customer_id.setdefault(cid, []).append(o)

        self.vouchers_by_customer_id: dict[str, list[dict]] = {}
        for v in self.vouchers:
            cid = v["customer_id"]
            self.vouchers_by_customer_id.setdefault(cid, []).append(v)

    def get_customer_by_id(self, customer_id: str) -> dict[str, Any]:
        """Tra cứu thông tin khách hàng."""
        customer = self.customer_by_id.get(customer_id)
        if not customer:
            return {"status": "not_found", "customer_id": customer_id}
        return {"status": "ok", "customer": customer}

    def get_orders_by_customer_id(self, customer_id: str, limit: int = 10) -> dict[str, Any]:
        """Lấy danh sách đơn hàng của khách hàng, mới nhất trước."""
        orders = self.orders_by_customer_id.get(customer_id)
        if not orders:
            return {"status": "not_found", "customer_id": customer_id}
        sorted_orders = sorted(orders, key=lambda o: o.get("created_at", ""), reverse=True)
        return {"status": "ok", "orders": sorted_orders[:limit]}

    def get_order_detail_by_order_id(self, order_id: str) -> dict[str, Any]:
        """Tra cứu chi tiết đơn hàng."""
        order = self.order_by_id.get(order_id)
        if not order:
            return {"status": "not_found", "order_id": order_id}
        return {"status": "ok", "order": order}

    def get_vouchers_by_customer_id(self, customer_id: str, only_active: bool = False) -> dict[str, Any]:
        """Tra cứu voucher của khách hàng. only_active=True chỉ lấy voucher còn dùng được."""
        vouchers = self.vouchers_by_customer_id.get(customer_id)
        if not vouchers:
            return {"status": "not_found", "customer_id": customer_id}
        if only_active:
            vouchers = [
                v
                for v in vouchers
                if v.get("status") == "active" and v.get("remaining_uses", 0) > 0
            ]
        return {"status": "ok", "total": len(vouchers), "vouchers": vouchers}


def build_data_tools(store: ShoppingDataStore) -> list:
    """Wrap 4 methods thành 4 LangChain tools nhỏ, riêng biệt."""

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
        """Lấy danh sách voucher của khách hàng. Nếu only_active=True thì chỉ lấy voucher còn dùng được (status=active và remaining_uses>0)."""
        return store.get_vouchers_by_customer_id(customer_id, only_active)

    return [
        get_customer_by_id,
        get_orders_by_customer_id,
        get_order_detail_by_order_id,
        get_vouchers_by_customer_id,
    ]
