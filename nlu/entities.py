"""Pulls order numbers out of free text and looks them up in the mock order
table, so "where is order 10234" gets a personal answer instead of a generic FAQ."""

import json
import re

import config

# Speech recognizers often return "1 0 2 3 4" for a spoken number, so digit
# groups separated by spaces are joined before extraction.
_SPACED_DIGITS = re.compile(r"(?<=\d)[\s-]+(?=\d)")
_ORDER_ID = re.compile(r"\b(?:ord(?:er)?[\s#:-]*)?(?:no\.?|number|id)?[\s#:]*(\d{5,8})\b", re.I)


def extract_order_id(text: str) -> str | None:
    m = _ORDER_ID.search(_SPACED_DIGITS.sub("", text))
    return m.group(1) if m else None


class OrderLookup:
    def __init__(self, orders: dict | None = None):
        if orders is None:
            with open(config.ORDERS_PATH, encoding="utf-8") as f:
                orders = json.load(f)
        self.orders = orders

    def describe(self, order_id: str) -> str | None:
        order = self.orders.get(order_id)
        if order is None:
            return None
        return f"Your order {order_id} ({order['item']}) is {order['status']}. {order['detail']}"
