"""Standard food names and categories for wine pairings.

Sources name foods slightly differently. ``normalize_food`` fixes known spelling
or translation quirks (X-Wines' "Maturated Cheese" -> "Aged Cheese") and
``food_category`` puts each food in one broad group, so a recommender can match
"cheese" to every cheese. Foods not listed here keep their (cleaned) name and get
no category: they are stored, never guessed.
"""

from __future__ import annotations

from typing import Dict, Optional

from .normalize import clean_text, fold

# Spelling / translation variants -> standard name.
FOOD_ALIASES: Dict[str, str] = {
    "maturated cheese": "Aged Cheese",
    "aged cheese": "Aged Cheese",
    "medium cured cheese": "Medium-cured Cheese",
    "game": "Game Meat",
    "souffle": "Soufflé",
    "yakisoba": "Yakisoba",
    "yakissoba": "Yakisoba",
    # Vivino's pairing labels
    "game deer venison": "Game Meat",
    "rich fish salmon tuna etc": "Rich Fish",
    "mild and soft cheese": "Soft Cheese",
    "mature and hard cheese": "Aged Cheese",
    "goat s milk cheese": "Goat Cheese",
    "appetizers and snacks": "Appetizer",
    "sweet desserts": "Sweet Dessert",
    "spicy food": "Spicy Food",
    "lean fish": "Lean Fish",
}

FOOD_CATEGORIES: Dict[str, str] = {}
for _category, _foods in {
    "Meat": [
        "Beef", "Lamb", "Veal", "Game Meat", "Pork", "Meat", "Barbecue",
        "Roast", "Grilled", "Light Stews",
    ],
    "Poultry": ["Poultry", "Chicken", "Duck", "Curry Chicken"],
    "Cured Meat": ["Cured Meat", "Ham", "Cold Cuts"],
    "Fish & Seafood": [
        "Rich Fish", "Lean Fish", "Fish", "Shellfish", "Seafood", "Codfish",
        "Sushi", "Sashimi",
    ],
    "Cheese": [
        "Cheese", "Soft Cheese", "Aged Cheese", "Hard Cheese", "Goat Cheese",
        "Blue Cheese", "Mild Cheese", "Medium-cured Cheese",
    ],
    "Vegetarian": [
        "Vegetarian", "Mushrooms", "Salad", "Beans", "Eggplant Parmigiana",
        "Baked Potato", "French Fries", "Tomato Dishes", "Chestnut",
    ],
    "Pasta & Grains": [
        "Pasta", "Pizza", "Risotto", "Lasagna", "Tagliatelle", "Paella", "Yakisoba",
    ],
    "Spicy & Asian": ["Spicy Food", "Asian Food"],
    "Appetizers & Snacks": ["Appetizer", "Snack", "Aperitif"],
    "Dessert": [
        "Dessert", "Sweet Dessert", "Fruit Dessert", "Citric Dessert", "Cake",
        "Cookies", "Chocolate", "Fruit", "Dried Fruits", "Spiced Fruit Cake",
        "Cream", "Soufflé",
    ],
}.items():
    for _food in _foods:
        FOOD_CATEGORIES[fold(_food)] = _category

def normalize_food(value) -> Optional[str]:
    """Clean a food name and apply known aliases. '' / None -> None."""
    text = clean_text(value)
    if not text:
        return None
    key = fold(text)
    return FOOD_ALIASES.get(key) or text


def food_category(name: Optional[str]) -> Optional[str]:
    return FOOD_CATEGORIES.get(fold(name)) if name else None
